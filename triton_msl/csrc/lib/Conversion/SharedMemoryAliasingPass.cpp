// ===-- SharedMemoryAliasingPass.cpp - reuse shared memory --------====//
//
// Post-lowering LLVM IR pass: coalesce addrspace(3) globals when their
// live ranges don't overlap. Simple greedy graph coloring.
//
// Runs after MLIR -> LLVM IR translation, before typed-pointer conversion.
//
// ===------------------------------------------------------------===//

#include "triton_msl/Conversion/TritonMSLToLLVM.h"

#include "llvm/IR/Module.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/CFG.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <limits>

namespace mlir {
namespace triton_msl {

// Loop-aware live-range extension (Option B).
//
// Our live-range computation uses linear instruction numbering. With a
// back-edge, a value defined before a loop and re-used on later iterations
// has a true live range that extends past its last textual use. Naively
// merging globals using the textual range can wrongly merge a pre-loop
// buffer (e.g. `q`) with a buffer stored inside the loop (e.g. `p`),
// producing silently wrong results on iterations 2+.
//
// FlashAttention's K-loop is the canonical offender — see compiler.py::
// _has_complex_ops for historical context on this bug.
//
// Fix: detect back-edges, group BBs into loops, and after computing
// textual live ranges, extend any global whose range intersects a loop
// span to cover the entire loop span. That way any pre-loop def used
// inside the loop is forced to conflict with any loop-body def, and
// aliasing will never merge them.

// Detect back-edges using linear block ordering: any pred that appears
// at or after the current BB's index is part of a back-edge. Returns a
// list of (header, latch) pairs of linear indices. Conservative but sound.
struct LoopSpan {
  unsigned headerIdx;  // linear block index of header
  unsigned latchIdx;   // linear block index of latch (pred via back-edge)
};

static llvm::SmallVector<LoopSpan, 4> findBackEdges(
    llvm::Function &F,
    const llvm::DenseMap<llvm::BasicBlock *, unsigned> &blockOrder) {
  llvm::SmallVector<LoopSpan, 4> edges;
  for (auto &BB : F) {
    auto hdrIt = blockOrder.find(&BB);
    if (hdrIt == blockOrder.end()) continue;
    unsigned thisIdx = hdrIt->second;
    for (auto *Pred : llvm::predecessors(&BB)) {
      auto predIt = blockOrder.find(Pred);
      if (predIt == blockOrder.end()) continue;
      if (predIt->second >= thisIdx) {
        edges.push_back({thisIdx, predIt->second});
      }
    }
  }
  return edges;
}

// Runs on an LLVM Module. Coalesces addrspace(3) globals where live ranges
// don't overlap. Call after LLVM IR generation, before typed-ptr conversion.
void aliasSharedMemoryGlobals(llvm::Module &mod) {
  // Collect all addrspace(3) globals that we generated.
  llvm::SmallVector<llvm::GlobalVariable *, 8> tgGlobals;
  for (auto &G : mod.globals()) {
    if (G.getAddressSpace() != 3) continue;
    auto name = G.getName();
    if (!name.starts_with("__tg_shared_") &&
        !name.starts_with("__reduce_shared_") &&
        !name.starts_with("__reduce2d_shared_") &&
        !name.starts_with("__tg_dot_out_")) continue;
    tgGlobals.push_back(&G);
  }
  if (tgGlobals.size() < 2) return;

  // For each function, compute liveness and build interference graph.
  for (auto &F : mod) {
    if (F.isDeclaration()) continue;

    // First, number blocks linearly and compute each block's [first, last]
    // instruction index span. We also collect back-edges to identify loops.
    llvm::DenseMap<llvm::BasicBlock *, unsigned> blockOrder;
    llvm::DenseMap<llvm::BasicBlock *, std::pair<int, int>> blockInstrSpan;
    unsigned bidx = 0;
    int instrIdx = 0;
    for (auto &BB : F) {
      blockOrder[&BB] = bidx++;
      int firstIdx = instrIdx;
      for (auto &I : BB) { (void)I; instrIdx++; }
      int lastIdx = instrIdx - 1;
      if (lastIdx < firstIdx) lastIdx = firstIdx;  // empty BB safety
      blockInstrSpan[&BB] = {firstIdx, lastIdx};
    }

    // Identify loop spans as contiguous linear block ranges
    // [headerIdx, latchIdx]. Convert to instruction index ranges.
    auto backEdges = findBackEdges(F, blockOrder);
    llvm::SmallVector<std::pair<int, int>, 4> loopInstrSpans;
    if (!backEdges.empty()) {
      // Build reverse map: block index -> BasicBlock*
      std::vector<llvm::BasicBlock *> blockByIdx(bidx, nullptr);
      for (auto &kv : blockOrder) blockByIdx[kv.second] = kv.first;
      for (const auto &e : backEdges) {
        // Find the min firstIdx and max lastIdx over blocks in
        // [headerIdx, latchIdx] (conservative: assume all blocks in that
        // linear range are part of the loop body).
        int lo = std::numeric_limits<int>::max();
        int hi = std::numeric_limits<int>::min();
        for (unsigned k = e.headerIdx; k <= e.latchIdx && k < bidx; ++k) {
          auto *BB = blockByIdx[k];
          if (!BB) continue;
          auto sp = blockInstrSpan[BB];
          if (sp.first < lo) lo = sp.first;
          if (sp.second > hi) hi = sp.second;
        }
        if (lo <= hi) loopInstrSpans.push_back({lo, hi});
      }
    }

    // Compute textual live ranges of globals as before.
    std::map<llvm::GlobalVariable *, std::pair<int, int>> liveRanges;
    instrIdx = 0;
    for (auto &BB : F) {
      for (auto &I : BB) {
        for (auto &Op : I.operands()) {
          if (auto *GV = llvm::dyn_cast<llvm::GlobalVariable>(Op.get())) {
            if (std::find(tgGlobals.begin(), tgGlobals.end(), GV)
                != tgGlobals.end()) {
              auto &range = liveRanges[GV];
              if (range.first == 0 && range.second == 0) {
                range.first = instrIdx;
                range.second = instrIdx;
              } else {
                range.second = instrIdx;
              }
            }
          }
        }
        instrIdx++;
      }
    }

    // Extend live ranges across loop spans: if a global crosses a loop
    // boundary (defined before and used inside, or defined inside and used
    // after) it must be live across the entire loop span — the back-edge
    // re-enters the loop and re-reads the value on later iterations.
    //
    // Globals whose entire textual range is strictly INSIDE a loop span
    // are considered loop-local (a single-iteration buffer) and are NOT
    // extended — they can legitimately alias with other loop-local buffers.
    // This is an unsound simplification (a loop-local global could in
    // principle be read on iteration k before it's (re)written on iteration
    // k+1), but in practice our shared-memory buffers follow a strict
    // write-then-read pattern inside each iteration, so loop-local buffers
    // do not interfere with each other across iterations.
    for (auto &kv : liveRanges) {
      auto &r = kv.second;
      for (const auto &sp : loopInstrSpans) {
        bool overlaps = (r.first <= sp.second) && (sp.first <= r.second);
        if (!overlaps) continue;
        bool crossesBoundary =
            (r.first < sp.first) || (r.second > sp.second);
        if (!crossesBoundary) continue;  // loop-local global, don't extend
        if (sp.first < r.first) r.first = sp.first;
        if (sp.second > r.second) r.second = sp.second;
      }
    }

    if (liveRanges.size() < 2) continue;

    // Optional debug dump (TRITON_MSL_SHMEM_DEBUG=1).
    bool shmemDebug = false;
    if (const char *d = std::getenv("TRITON_MSL_SHMEM_DEBUG")) {
      shmemDebug = (d[0] == '1');
    }
    if (shmemDebug) {
      llvm::errs() << "[shmem-alias] fn=" << F.getName()
                   << " loops=" << loopInstrSpans.size() << "\n";
      for (const auto &sp : loopInstrSpans) {
        llvm::errs() << "  loop span [" << sp.first << "," << sp.second
                     << "]\n";
      }
      for (auto &kv : liveRanges) {
        llvm::errs() << "  " << kv.first->getName() << " ["
                     << kv.second.first << "," << kv.second.second << "]\n";
      }
    }

    // Build interference graph.
    std::vector<llvm::GlobalVariable *> globals;
    for (auto &kv : liveRanges) globals.push_back(kv.first);
    int n = globals.size();
    std::vector<std::set<int>> adj(n);
    for (int i = 0; i < n; ++i) {
      for (int j = i + 1; j < n; ++j) {
        auto &ri = liveRanges[globals[i]];
        auto &rj = liveRanges[globals[j]];
        // Overlap: [ri.first, ri.second] intersects [rj.first, rj.second]
        if (ri.first <= rj.second && rj.first <= ri.second) {
          adj[i].insert(j);
          adj[j].insert(i);
          continue;
        }
        // Never merge globals with different value types: downstream
        // typed-pointer conversion preserves the original GEP element
        // types, and mixing e.g. [32 x float] with [1024 x float] under
        // the same symbol yields "defined with type X but expected Y"
        // errors from Metal's non-opaque-pointer compiler.
        if (globals[i]->getValueType() != globals[j]->getValueType()) {
          adj[i].insert(j);
          adj[j].insert(i);
        }
      }
    }

    // Greedy color by size (largest first).
    std::vector<int> order(n);
    for (int i = 0; i < n; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
      auto sizeA = mod.getDataLayout().getTypeAllocSize(
          globals[a]->getValueType());
      auto sizeB = mod.getDataLayout().getTypeAllocSize(
          globals[b]->getValueType());
      return sizeA > sizeB;
    });

    std::vector<int> color(n, -1);
    // For each color, track the index of the member with the largest
    // allocation size. We reuse that member's value type as the merged
    // global's type so existing GEPs remain type-consistent (e.g. all
    // members are `[32 x float]` -> merged stays `[32 x float]`). This
    // avoids downstream type mismatches in the typed-pointer post-pass.
    std::map<int, int> colorRepIdx;
    std::map<int, uint64_t> colorSize;
    for (int idx : order) {
      std::set<int> used;
      for (int nb : adj[idx]) {
        if (color[nb] != -1) used.insert(color[nb]);
      }
      int c = 0;
      while (used.count(c)) c++;
      color[idx] = c;
      auto sz = mod.getDataLayout().getTypeAllocSize(
          globals[idx]->getValueType());
      auto it = colorSize.find(c);
      if (it == colorSize.end() || sz > it->second) {
        colorSize[c] = sz;
        colorRepIdx[c] = idx;
      }
    }

    // Count unique colors; if all colors are unique (no aliasing possible),
    // skip to avoid churning the IR.
    std::set<int> uniqueColors(color.begin(), color.end());
    if ((int)uniqueColors.size() == n) continue;

    if (shmemDebug) {
      for (int i = 0; i < n; ++i) {
        llvm::errs() << "  color " << globals[i]->getName()
                     << " -> " << color[i] << "\n";
      }
    }

    // Create merged globals, one per color. Use the representative member's
    // value type (it is the largest in the group), so the merged global's
    // declared type matches the GEP element types already in the IR.
    std::map<int, llvm::GlobalVariable *> colorToMerged;
    for (auto &[c, sz] : colorSize) {
      auto *rep = globals[colorRepIdx[c]];
      auto *valTy = rep->getValueType();
      std::string name = "__tg_merged_" + std::to_string(c);
      auto *merged = new llvm::GlobalVariable(
          mod, valTy, /*isConstant=*/false,
          llvm::GlobalValue::InternalLinkage,
          llvm::UndefValue::get(valTy), name, /*InsertBefore=*/nullptr,
          llvm::GlobalValue::NotThreadLocal, /*AddressSpace=*/3);
      merged->setAlignment(llvm::MaybeAlign(16));
      colorToMerged[c] = merged;
    }

    // Replace original globals with the merged global (or a bitcast to the
    // original's pointer type if their value types differ). Pointer types
    // in opaque-pointer mode are identical (just `ptr addrspace(3)`), so
    // `replaceAllUsesWith` usually works directly; the bitcast fallback
    // keeps us correct if a legacy typed-pointer consumer sneaks in.
    for (int i = 0; i < n; ++i) {
      auto *orig = globals[i];
      auto *merged = colorToMerged[color[i]];
      if (orig->getType() == merged->getType()) {
        orig->replaceAllUsesWith(merged);
      } else {
        orig->replaceAllUsesWith(
            llvm::ConstantExpr::getBitCast(merged, orig->getType()));
      }
      orig->eraseFromParent();
    }
  }
}

} // namespace triton_msl
} // namespace mlir

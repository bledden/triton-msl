# fp16/bf16 atomic RMW via word-CAS — design (2026-06-13)

> Lift the loud refusal on 16-bit float atomics (`_lower_atomic_rmw`,
> `_lowerer_control.py:586`) by emitting a neighbor-preserving 32-bit word-CAS.
> Closes ~298 `test_atomic_rmw[...float16/bfloat16...]` skips — Phase 3 feature 1.
> Python/MSL per the 2026-06-11 language decision.

## Problem
Metal has no 16-bit device atomic. The existing float-atomic path (`fp32`) does a
CAS loop on `atomic_uint*` reinterpreting the 32-bit slot. For an fp16/bf16
element that reads/writes 4 bytes over a 2-byte value — corrupting it AND its
neighbor — so the lowerer currently refuses loudly (correct, but a gap).

## Approach — word-CAS with half-splice
A 16-bit element lives in one half of an aligned 4-byte word. Atomically RMW it by
CAS-looping on the containing word and preserving the other half:

```
elem_index = offsets            # in 16-bit element units (existing addressing)
word_ptr   = (device atomic_uint*)base_ptr + (elem_index >> 1)   # 4-byte-aligned word
shift      = 16u * (uint)(elem_index & 1)                        # little-endian: half 0 = low bits
w = atomic_load_explicit(word_ptr, relaxed)
while (true):
    ushort cur_bits = (ushort)((w >> shift) & 0xFFFFu)
    HALF   old      = as_type<HALF>(cur_bits)        # HALF ∈ {half, bfloat}
    HALF   new      = OP(old, (HALF)val)             # add / max / min / exch
    ushort new_bits = as_type<ushort>(new)
    uint   w_new    = (w & ~(0xFFFFu << shift)) | ((uint)new_bits << shift)
    if atomic_compare_exchange_weak_explicit(word_ptr, &w, w_new, relaxed, relaxed): break
result = as_type<HALF>(cur_bits)   # the OLD value (Triton atomic returns prior)
```

- **Alignment** is guaranteed: Metal buffer bindings are ≥16-byte aligned and the
  `tt.divisibility=16` arg hint holds; `base + (i>>1)` words are always 4-byte
  aligned. No runtime alignment guard needed.
- **Endianness:** Apple GPU is little-endian → element `i&1==0` is the low 16 bits.
- **HALF type:** `half` for fp16; `bfloat` for bf16 (native on Metal 3.1+/macOS 14+,
  fine on M4 Max). Documented manual fallback for bf16 if a toolchain lacks
  `bfloat`: bf16→float `as_type<float>((uint)bits << 16)`, float→bf16
  round-to-nearest-even truncate of the top 16 bits.
- **OP:** `add`/`fadd` → `old + v`; `max` → `max(old, v)`; `min` → `min(old, v)`;
  `exch` → `v` (no read needed, but the splice still preserves the neighbor, so it
  still uses the CAS loop rather than a bare `atomic_exchange` on the word).

## Components (files)
- `triton_metal/codegen/_lowerer_control.py` — `_lower_atomic_rmw`: replace the
  16-bit refusal (586–600) with a `_emit_word_cas_16bit(...)` branch wired into the
  op dispatch; result type becomes the element type (fp16/bf16). The fp32/int paths
  are unchanged.
- `scripts/conftest_metal.py` — remove the now-passing `test_atomic_rmw` fp16/bf16
  skip entries (the ratchet moves UP).
- Tests: a project GPU test (`tests/test_fp16_atomics.py`) for correctness, plus the
  un-skipped upstream `test_atomic_rmw` family as the corpus gate.

## Error handling / integrity
- Ops outside {add, fadd, max, min, exch} on a 16-bit float still refuse loudly
  (`MetalNonRecoverableError`) rather than emit untested code.
- The CAS loop is the standard lock-free RMW; correctness rests on neighbor
  preservation (the `& ~(0xFFFF<<shift)` mask) — covered by a test that writes
  *adjacent* fp16 elements from different threads and checks neither corrupts the
  other.
- `TRITON_METAL_MEPT=0` is unaffected (this is in the atomic path, not MEPT-gated).

## Testing / ratchet
- **Unit/correctness (GPU, serial):** fp16 and bf16 `atomic_add` accumulating many
  threads into one slot (sum correctness); fp16 `atomic_max`/`atomic_min`; a
  **neighbor-preservation** test (threads atomically add to even/odd adjacent fp16
  slots; both correct). bf16 tolerance accounts for its 8-bit mantissa.
- **Corpus:** un-skip the upstream `test_atomic_rmw[...float16...]`/`[...bfloat16...]`
  family in conftest_metal; they must pass. Upstream `test_core` count moves UP
  (~+298), never down — the ratchet rule.
- **Regression:** flag-default project suite stays green; the fp32/int atomic tests
  (already passing) unaffected.

## Out of scope
- 64-bit (i64/u64) atomics — hardware-impossible (no Metal 64-bit device atomic).
- fp16 atomic_cas (`tt.atomic_cas`) if not in the `test_atomic_rmw` family — only
  extend `_lower_atomic_cas` if the un-skipped corpus needs it; otherwise leave its
  refusal.

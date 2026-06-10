"""Per-family op coverage for the C++ MLIR path (Phase 1 spec)."""

FAMILIES = {
    "elementwise": {
        # -- Triton ops (custom patterns in ElementwiseOpToLLVM.cpp) --
        'tt.get_program_id', 'tt.get_num_programs',
        'tt.make_range', 'tt.splat', 'tt.broadcast',
        'tt.expand_dims', 'tt.reshape',
        'tt.addptr', 'tt.load', 'tt.store',
        'tt.func', 'tt.return',
        'tt.reduce', 'tt.reduce.return',
        'tt.extern_elementwise',
        'tt.dot',
        # -- Arith ops (standard arith-to-LLVM + custom constant) --
        'arith.constant',
        'arith.addf', 'arith.addi',
        'arith.subf', 'arith.subi',
        'arith.mulf', 'arith.muli',
        'arith.divf', 'arith.divsi', 'arith.divui',
        'arith.remf', 'arith.remsi', 'arith.remui',
        'arith.negf',
        'arith.andi', 'arith.ori', 'arith.xori',
        'arith.shli', 'arith.shrsi', 'arith.shrui',
        'arith.cmpi', 'arith.cmpf',
        'arith.select',
        'arith.sitofp', 'arith.fptosi',
        'arith.uitofp', 'arith.fptoui',
        'arith.extf', 'arith.truncf',
        'arith.extsi', 'arith.extui', 'arith.trunci',
        'arith.bitcast', 'arith.index_cast',
        'arith.maxnumf', 'arith.minnumf',
        'arith.maximumf', 'arith.minimumf',
        'arith.maxsi', 'arith.minsi',
        'arith.maxui', 'arith.minui',
        # -- Math ops (standard math-to-LLVM) --
        'math.exp', 'math.exp2',
        'math.log', 'math.log2', 'math.log10',
        'math.sqrt', 'math.rsqrt',
        'math.absf', 'math.abs',
        'math.sin', 'math.cos', 'math.tan',
        'math.tanh', 'math.erf',
        'math.ceil', 'math.floor', 'math.round',
        'math.powf', 'math.fma',
        'math.copysign',
        # -- Control flow (standard cf-to-LLVM) --
        'cf.br', 'cf.cond_br',
        # -- SCF (structured control flow) --
        'scf.for', 'scf.yield', 'scf.if',
        # -- TritonGPU shared memory ops (handled by C++ path) --
        'ttg.local_alloc', 'ttg.local_load', 'ttg.local_store',
        'ttg.local_dealloc',
        'ttg.memdesc_subview', 'ttg.memdesc_trans',
        'ttg.async_copy_global_to_local', 'ttg.async_wait',
        'ttg.convert_layout',
    },
}

ENABLED = {"elementwise"}


def enabled_ops():
    """Union of allowed ops across all enabled families."""
    out = set()
    for fam in ENABLED:
        out |= FAMILIES[fam]
    return out

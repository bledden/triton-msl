"""scf.while carried-variable correctness with multi-comparison conditions.

A `while (a<n) and (cond):` loop whose body modifies a carried var (so the
condition's `cond` term stays live) mis-lowered: the parser keyed cmpi
predicates by SSA name, and scf.while's before/after regions reuse names
(%0/%1), so the after-region predicate overwrote the before-region's. The
`a<n` term emitted as `a>=n` -> the loop broke immediately and the carried
var read its initial value (downstream tridec bug 3, 2026-06-10).
"""
import pytest

try:
    import torch
    import triton
    import triton.language as tl
    import Metal
    HAS = Metal.MTLCreateSystemDefaultDevice() is not None
except Exception:
    HAS = False

requires_metal = pytest.mark.skipif(not HAS, reason="Metal/torch/triton needed")

if HAS:
    @triton.jit
    def _while_early_exit(O, n_legs):
        leg = 0; done = 0
        while (leg < n_legs) and (done == 0):
            leg += 1
            if leg >= 3:
                done = 1
        tl.store(O, leg)


@requires_metal
def test_while_carried_var_early_exit():
    # Loop should run leg 1,2,3, set done at leg=3, exit -> leg == 3.
    o = torch.zeros(1, dtype=torch.int32)
    _while_early_exit[(1,)](o, 10)
    assert int(o[0]) == 3, f"carried leg wrong: {int(o[0])} (0 => loop broke immediately)"

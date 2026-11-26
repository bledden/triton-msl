"""A dot with constant operands is lowered by a template only where the template is the source.

Two templates sat behind the loop-free dot routes with no proof of what they bind or store:

* A dot of two constant tiles took a template that stores the whole tile densely
  (``C[i * N + j]``) from every program, through the LAST pointer argument. A store that was
  masked, tiled by ``program_id``, strided or padded was overwritten: on the GPU a store
  clipped at 20 rows wrote 32, a second row tile was never written, a row stride of 50 wrote
  the dense cells. The storage proof judges the source's stores, so an output of exactly those
  cells passed it and the shader then wrote past it.
* A dot with ONE constant operand has two pointer arguments. It reached the positional matmul
  template (buffers 0/1/2 are A, B, C; 3/4/5 are M, N, K), whose role guard ran only for three
  or more pointers. The shader read the output as B and stored through the buffer of the
  scalar argument at slot 2.

Both now refuse at compile unless the template is exactly the source. The second kind is never
launched here: these tests read the lowering's decision and the emitted shader text. The last
tests launch each template the storage proof covers inside storages they own and compare the
whole output storage with the source, which is the check that a template stores exactly the
cells its proof is about.
"""

import numpy as np
import pytest
import torch
import triton
import triton.language as tl

from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton._C.libtriton import ir

from tests.test_a_loop_free_dot_proves_its_storage import (
    FILL,
    LAUNCHES,
    PLAN_LAUNCHES,
    REFUSED,
    OUTSIDE,
    _expected,
    _install_tripwire,
    _forget_tripped_lease,
    _storages,
)
from triton_msl.backend.compiler import MetalBackend
from triton_msl.codegen.msl_emitter import emit_msl
from triton_msl.errors import MetalNonRecoverableError

requires_gpu = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal GPU needed")


@triton.jit
def _both_constant(
    C, X, M, N, SC, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, FORM: tl.constexpr, LC: tl.constexpr
):
    # FORM: 0 dense unmasked; 1 tiled by program_id; 2 clipped at runtime M, N; 3 runtime row
    # stride; 4 literal padded row stride LC; 5 clipped at a constant below the tile.
    rm = tl.arange(0, BM)
    rn = tl.arange(0, BN)
    if FORM == 1:
        rm = tl.program_id(0) * BM + rm
    c = tl.dot(tl.full((BM, BK), 1.0, tl.float32), tl.full((BK, BN), 2.0, tl.float32))
    if FORM == 3:
        cp = C + rm[:, None] * SC + rn[None, :]
    elif FORM == 4:
        cp = C + rm[:, None] * LC + rn[None, :]
    else:
        cp = C + rm[:, None] * BN + rn[None, :]
    if FORM == 2:
        tl.store(cp, c, mask=(rm[:, None] < M) & (rn[None, :] < N))
    elif FORM == 5:
        tl.store(cp, c, mask=rm[:, None] < 20)
    else:
        tl.store(cp, c)


@triton.jit
def _constant_b(A, C, M, N, SA, SC, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, TILED: tl.constexpr):
    rm = tl.arange(0, BM)
    if TILED:
        rm = tl.program_id(0) * BM + rm
    rn = tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a = tl.load(A + rm[:, None] * SA + rk[None, :])
    c = tl.dot(a, tl.full((BK, BN), 2.0, tl.float32))
    tl.store(C + rm[:, None] * SC + rn[None, :], c, mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def _constant_a(B, C, M, N, SB, SC, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, TILED: tl.constexpr):
    rm = tl.arange(0, BM)
    if TILED:
        rm = tl.program_id(0) * BM + rm
    rn = tl.arange(0, BN)
    rk = tl.arange(0, BK)
    b = tl.load(B + rk[:, None] * SB + rn[None, :])
    c = tl.dot(tl.full((BM, BK), 1.0, tl.float32), b)
    tl.store(C + rm[:, None] * SC + rn[None, :], c, mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def _constant_b_three_pointers(A, X, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # Three pointers and three integers in the template's slots, but X is not an operand.
    rm = tl.arange(0, BM)
    rn = tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a = tl.load(A + rm[:, None] * K + rk[None, :])
    c = tl.dot(a, tl.full((BK, BN), 2.0, tl.float32))
    tl.store(C + rm[:, None] * N + rn[None, :], c)


@triton.jit
def _constant_to_half(H, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # Upstream test_dot_without_load's float16 form: the product is cast once on its way to the store.
    rm = tl.arange(0, BM)
    rn = tl.arange(0, BN)
    c = tl.dot(tl.full((BM, BK), 1.0, tl.float16), tl.full((BK, BN), 1.0, tl.float16))
    tl.store(H + rm[:, None] * BN + rn[None, :], c.to(tl.float16))


def _lower(kernel, constants):
    backend = MetalBackend(GPUTarget("metal", "apple-m4", 32))
    options = backend.parse_options({"num_warps": 4})
    pointers = ("A", "B", "C", "X")
    signature = {
        name: ("constexpr" if name in constants else "*fp16" if name == "H" else "*fp32" if name in pointers else "i32")
        for name in kernel.arg_names
    }
    context = ir.context()
    ir.load_dialects(context)
    module = ASTSource(kernel, signature, constants).make_ir(
        backend.target, options, backend.get_codegen_implementation(options), backend.get_module_map(), context
    )
    module = backend.make_ttir(module, {}, options)
    module = backend.make_ttgir(module, {}, options)
    metadata = {}
    return emit_msl(module, metadata, options), metadata


BLOCKS = dict(BM=32, BN=32, BK=32)
POSITIONAL = "volatile device"  # the positional matmul template declares its output this way


@pytest.mark.parametrize("block", [16, 32])
def test_a_dense_constant_dot_stores_through_the_pointer_it_addresses(block):
    """The store's own pointer (argument 0), not the last pointer argument (X, argument 1)."""
    msl, _ = _lower(_both_constant, dict(BM=block, BN=block, BK=block, FORM=0, LC=block))
    assert f"_mmarg0[_i * {block}u + _j] = " in msl
    assert "_mmarg1[" not in msl


def test_a_dense_constant_dot_keeps_its_one_output_cast():
    msl, _ = _lower(_constant_to_half, BLOCKS)
    assert "_mmarg0[_i * 32u + _j] = (half)_sum;" in msl


@triton.jit
def _constant_cast_chain(C, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, CHAIN: tl.constexpr):
    # CHAIN: the product rounded through float16 on its way to a float32 output.
    rm = tl.arange(0, BM)
    rn = tl.arange(0, BN)
    c = tl.dot(tl.full((BM, BK), 0.1, tl.float32), tl.full((BK, BN), 0.3, tl.float32))
    if CHAIN:
        c = c.to(tl.float16).to(tl.float32)
    tl.store(C + rm[:, None] * BN + rn[None, :], c)


def test_a_constant_dot_rounded_through_another_type_is_refused():
    """The template converts once; a cast round trip is a value it does not compute."""
    msl, _ = _lower(_constant_cast_chain, dict(BLOCKS, CHAIN=False))
    assert "_mmarg0[_i * 32u + _j] = (float)_sum;" in msl
    with pytest.raises(MetalNonRecoverableError):
        _lower(_constant_cast_chain, dict(BLOCKS, CHAIN=True))


@pytest.mark.parametrize("form", [1, 2, 3, 4, 5])
def test_a_constant_dot_the_template_would_overwrite_is_refused(form):
    """Tiled, clipped, strided or padded: the dense whole-tile template is another program."""
    with pytest.raises(MetalNonRecoverableError) as refusal:
        _lower(_both_constant, dict(BLOCKS, FORM=form, LC=50))
    assert "Refusing" in str(refusal.value)


@pytest.mark.parametrize("kernel", [_constant_a, _constant_b])
@pytest.mark.parametrize("tiled", [False, True])
@pytest.mark.parametrize("block", [16, 32])
def test_a_dot_with_one_constant_operand_never_takes_the_positional_template(kernel, tiled, block):
    """Two pointer arguments: nothing may bind them to the template's A/B/C/M/N/K slots."""
    try:
        msl, metadata = _lower(kernel, dict(BM=block, BN=block, BK=block, TILED=tiled))
    except MetalNonRecoverableError:
        return
    assert POSITIONAL not in msl and not metadata.get("fast_matmul")


def test_three_pointers_with_a_constant_operand_are_refused():
    with pytest.raises(MetalNonRecoverableError):
        _lower(_constant_b_three_pointers, BLOCKS)


# ---- every template the storage proof covers stores exactly the source's cells ------------------


def _constant_launch():
    from tests.test_a_loop_free_dot_proves_its_storage import Source

    class Constant(Source):
        def run(self, a_host, b_host, c_host, pointers):
            for _, _, cp, mask in self.programs:
                c_host[pointers[2] + cp[mask]] = 64.0  # 32 * 1.0 * 2.0

    # The unused pointer X is given a tile of its own, so that a shader which stored through it
    # (the last pointer argument, as this template once chose) would stay inside owned storage.
    source = Constant((32, 32, 32), (32, 1, 32, 1, 32, 1))
    return source, lambda a, b, c: _both_constant[(1,)](c, a, 32, 32, 32, FORM=0, LC=32, **BLOCKS)


# row -> (the product's name for the template, the launch)
TEMPLATES = {
    "dense": ("dense single tile", "dense"),
    "dense_trans_a": ("dense single tile", "dense_trans_a"),
    "scalar": ("scalar single tile", "loop_free"),
    "scalar_tiled_clipped": ("scalar single tile", "loop_free_partial_tiles"),
    "scalar_negative_stride": ("scalar single tile", "loop_free_negative_row_stride"),
    "loop_bound_one": ("scalar single tile", "loop_bound_one_partial_tiles"),
    "fused_softmax_clipped": ("fused softmax/epilogue", "softmax_32_clipped"),
    "generic_epilogue_clipped": ("generic dot", "epilogue_32_clipped"),
    "generic_masked_loads": ("generic dot", "masked_loads_loop_bound_one_two_tiles"),
    "constant": ("constant dot", None),
}


def test_every_template_with_a_storage_proof_has_a_row():
    """A template added to the product's list without a launch-and-compare row fails here."""
    from triton_msl.codegen.generic_lowerer import GenericLowerer

    assert {kind for kind, _ in TEMPLATES.values()} == set(GenericLowerer.STORAGE_PROVED_TEMPLATES)


def _launcher(name):
    target = TEMPLATES[name][1]
    if target is None:
        return _constant_launch()
    return (LAUNCHES.get(target) or PLAN_LAUNCHES[target])()


@requires_gpu
@pytest.mark.parametrize("compile_shader", ["1", "0"])
@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_a_template_stores_exactly_the_cells_its_proof_is_about(name, compile_shader, monkeypatch):
    """The whole output storage equals the source: no cell outside the proved store set changes.

    Then the proof is asked about an output that ends at the source's last stored cell (accepted)
    and one element short (refused), with the submission methods replaced, so the set the proof
    judges is shown to be the set the shader was just seen to write.
    """
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", compile_shader)
    source, launch = _launcher(name)
    hosts, pointers, tensors, views = _storages(source, "offset")
    expected = _expected(source, hosts, pointers)
    launch(*views)
    torch.mps.synchronize()
    got = tensors[2].cpu().numpy().astype(np.float64)
    same = np.isclose(got, expected, rtol=2e-5, atol=2e-6) if source.epilogue == "softmax" else got == expected
    stray = int(np.sum((got != FILL) & (expected == FILL)))
    assert bool(same.all()), f"{int((~same).sum())} cells differ from the source, {stray} of them outside its store set"
    assert np.array_equal(tensors[0].cpu().numpy(), hosts[0]) and np.array_equal(tensors[1].cpu().numpy(), hosts[1])
    for short, refused in ((None, False), ((2, "back"), True)):
        _, _, _, exact = _storages(source, "exact", short=short)
        with monkeypatch.context() as patch:
            hits = _install_tripwire(patch)
            try:
                launch(*exact)
                outcome = "dispatched"
            except MetalNonRecoverableError as refusal:
                outcome = "refused" if (REFUSED in str(refusal) or OUTSIDE in str(refusal)) else str(refusal)
            except Exception:
                outcome = "dispatched" if hits else "error"
            finally:
                _forget_tripped_lease(hits)
        assert outcome == ("refused" if refused else "dispatched")

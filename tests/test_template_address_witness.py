"""Every source address an attention template replays must be the template's own address.

The detectors read each pointer's strides by their POSITION in the address chain. That fixes
neither the index a stride multiplies, nor the batch coordinate, nor the loop advance, and the
template then rebuilds every address from its own coordinates. On the varlen, MLA and backward
routes ``_address_witness`` evaluates each source load/store address exactly on random integers,
requires it to equal the template's formula, and binds every stride by value (so an equivalent
respelling binds the stride it actually uses). The biased forward routes (2-D and 3-D) match
each address term structurally instead (``_prove_biased_fa_addresses``). The dense route has no
address proof of its own: its detector already refused these variants, and its cases here are
regression guards.

Sections: the witness itself on small kernels (no device); then, on the GPU, one-change variants
of the dense, biased (2-D and 3-D), varlen and MLA forward kernels. A variant either refuses
with a route-specific message or computes what the SOURCE computes, checked against a float64
reference evaluated from the variant's own address spelling into poisoned buffers. The backward
routes are in test_fa_bwd_address_witness.py.
"""

import math
import random

import pytest
import torch
import triton
import triton.language as tl
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from triton_msl.backend.compiler import MetalBackend
from triton_msl.codegen.mlir_walker import walk_ttgir
from triton_msl.errors import MetalNonRecoverableError

requires_mps = pytest.mark.skipif(
    not (torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")),
    reason="needs MPS + compile_shader",
)

POISON = -16384.0


def _variant(kernel, *edits):
    """A fresh JITFunction of ``kernel`` with each (old, new) applied exactly once."""
    fn = triton.JITFunction(kernel.fn)
    src = fn.src
    for old, new in edits:
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    fn._unsafe_update_src(src)
    return fn


def _graph(fn, constexprs, pointers):
    target = GPUTarget("metal", "apple-m4", 32)
    backend = MetalBackend(target)
    options = backend.parse_options({})
    pointers = pointers if isinstance(pointers, dict) else dict.fromkeys(pointers, "*fp32")
    signature = {n: pointers.get(n, "i32") for n in fn.arg_names if n not in constexprs}
    source = ASTSource(fn, signature, constexprs=constexprs)
    context = ir.context()
    ir.load_dialects(context)
    mod = source.make_ir(
        target, options, backend.get_codegen_implementation(options), backend.get_module_map(), context
    )
    mod = backend.make_ttir(mod, {}, options)
    mod = backend.make_ttgir(mod, {}, options)
    return walk_ttgir(mod, options)


# ---------------------------------------------------------------------------------------
# stride_value and _Env
# ---------------------------------------------------------------------------------------


def test_stride_value_reads_each_detector_form():
    from triton_msl.codegen._address_witness import _Env, stride_value

    env = _Env(random.Random(1), {}, has_loop=False)
    assert stride_value("c1", env) == 1
    assert stride_value("c0", env) == 0
    assert stride_value(("literal", 7), env) == 7
    assert stride_value(3, env) == env.scalar(3)
    assert stride_value(("mul", 3, ("literal", 2)), env) == 2 * env.scalar(3)
    assert stride_value(("mul", ("mul", 2, 3), "c1"), env) == env.scalar(2) * env.scalar(3)


@pytest.mark.parametrize(
    "expr",
    ["c2", None, True, 1.0, ("literal", 1.5), ("literal", True), ("literal",), ("mul", 3), ("add", 1, 2), [3]],
    ids=repr,
)
def test_stride_value_refuses_any_other_form(expr):
    from triton_msl.codegen._address_witness import WitnessRefusal, _Env, stride_value

    with pytest.raises(WitnessRefusal, match="a stride expression the template cannot bind"):
        stride_value(expr, _Env(random.Random(1), {}, has_loop=False))


def test_env_atoms():
    from triton_msl.codegen._address_witness import _Env

    scalars = {}
    rng = random.Random(5)
    a, b = _Env(rng, scalars, has_loop=True), _Env(rng, scalars, has_loop=True)
    # Arguments and the trip count are shared by every assignment drawn from one scalar table.
    assert a.scalar(4) == b.scalar(4) and a.atom(("trip",)) == b.atom(("trip",))
    assert 1 << 20 <= a.scalar(4) < 1 << 40
    # Coordinate atoms are drawn per assignment and memoised within it.
    assert a.atom(("pid", 0)) == a.atom(("pid", 0))
    assert 1 << 20 <= a.atom(("t",)) < 1 << 40
    assert _Env(rng, {}, has_loop=False).atom(("t",)) == 0
    zero = _Env(rng, scalars, has_loop=True, zero=True, ones=(("pid", 1),))
    assert (zero.atom(("pid", 0)), zero.atom(("pid", 1)), zero.atom(("t",))) == (0, 1, 0)
    assert zero.scalar(4) == a.scalar(4)
    # A quotient and a remainder by an argument derive from the one sampled program id, so an
    # address spelled with either satisfies the identity that ties them to it.
    arg, n = ("arg", 4), a.scalar(4)
    quotient, remainder = a.atom(("div", 1, arg)), a.atom(("rem", 1, arg))
    assert (quotient, remainder) == divmod(a.atom(("pid", 1)), n) and quotient >= 1 << 20
    # Raising the quotient to 1 puts the program id at the divisor; the remainder, at 1.
    one = _Env(rng, scalars, has_loop=True, zero=True, ones=(("div", 1, arg),))
    assert (one.atom(("div", 1, arg)), one.atom(("rem", 1, arg)), one.atom(("pid", 1))) == (1, 0, n)
    one = _Env(rng, scalars, has_loop=True, zero=True, ones=(("rem", 1, arg),))
    assert (one.atom(("div", 1, arg)), one.atom(("rem", 1, arg)), one.atom(("pid", 1))) == (0, 1, 1)
    # A constant divisor keeps free atoms: a remainder below a small constant would separate
    # two formulas too rarely.
    const = ("div", 1, ("const", 4))
    assert 1 << 20 <= a.atom(const) < 1 << 40 and zero.atom(const) == 0


# ---------------------------------------------------------------------------------------
# AddressWitness / prove_template_addresses on a small kernel (no device)
# ---------------------------------------------------------------------------------------


@triton.jit
def _tile(X, Y, sxb, sxh, sxr, sxc, syb, syh, syr, syc, NH, N, B: tl.constexpr, D: tl.constexpr):
    """Y[b, h, rows of program 0] = sum over the key loop of X[b, h, keys]; b, h = program 1 / NH, % NH."""
    pid = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // NH
    h = bh % NH
    r = pid * B + tl.arange(0, B)
    j = tl.arange(0, B)
    c = tl.arange(0, D)
    acc = tl.zeros([B, D], tl.float32)
    for t in range(0, N, B):
        acc += tl.load(X + b * sxb + h * sxh + (t + j)[:, None] * sxr + c[None, :] * sxc)
    tl.store(Y + b * syb + h * syh + r[:, None] * syr + c[None, :] * syc, acc)


_X = "tl.load(X + b * sxb + h * sxh + (t + j)[:, None] * sxr + c[None, :] * sxc)"
_Y = "tl.store(Y + b * syb + h * syh + r[:, None] * syr + c[None, :] * syc, acc)"
_TILE_CONST = dict(B=4, D=2)


def _tile_graph(*edits):
    return _graph(_variant(_tile, *edits), _TILE_CONST, ("X", "Y"))


def _tile_roles(graph):
    index = {a.name: a.index for a in graph.args}
    batch, head = ("div", 1, ("arg", index["NH"])), ("rem", 1, ("arg", index["NH"]))

    def rows(atom):
        return lambda env: [env.atom(atom) * 4 + i for i in range(4)]

    def role(name, ptr, kind, prefix, atom):
        return {
            "name": name,
            "arg": index[ptr],
            "kind": kind,
            # Positional, as a detector reads them: [batch, head, row, column].
            "strides": [index[prefix + s] for s in ("b", "h", "r", "c")],
            "shape": (4, 2),
            "scalars": [(0, batch), (1, head)],
            "axes": [(2, 0, rows(atom)), (3, 1, lambda env: [0, 1])],
        }

    roles = [role("X", "X", "load", "sx", ("t",)), role("Y", "Y", "store", "sy", ("pid", 0))]
    return roles, (index["N"], 4), index


def test_witness_evaluates_every_access_exactly():
    from triton_msl.codegen._address_witness import AddressWitness

    graph = _tile_graph()
    index = {a.name: a.index for a in graph.args}
    witness = AddressWitness(graph)
    env = witness.env(random.Random(3), {})
    (kx, bx, ox), (ky, by, oy) = witness.offsets(env)
    assert (kx, bx, ky, by) == ("load", index["X"], "store", index["Y"])
    s, nh = env.scalar, env.atom(("div", 1, ("arg", index["NH"])))
    hh = env.atom(("rem", 1, ("arg", index["NH"])))
    t = env.atom(("t",)) * 4  # the loop runs range(0, N, 4) from zero
    want_x = [
        nh * s(index["sxb"]) + hh * s(index["sxh"]) + (t + i) * s(index["sxr"]) + c * s(index["sxc"])
        for i in range(4)
        for c in range(2)
    ]
    r0 = env.atom(("pid", 0)) * 4
    want_y = [
        nh * s(index["syb"]) + hh * s(index["syh"]) + (r0 + i) * s(index["syr"]) + c * s(index["syc"])
        for i in range(4)
        for c in range(2)
    ]
    assert (ox.shape, ox.data) == ((4, 2), want_x)
    assert (oy.shape, oy.data) == ((4, 2), want_y)


def test_witness_proves_the_canonical_addresses():
    from triton_msl.codegen._address_witness import prove_template_addresses

    graph = _tile_graph()
    roles, loop, index = _tile_roles(graph)
    bound = prove_template_addresses(graph, roles, loop)
    assert bound == {
        "X": [index[n] for n in ("sxb", "sxh", "sxr", "sxc")],
        "Y": [index[n] for n in ("syb", "syh", "syr", "syc")],
    }


@pytest.mark.parametrize(
    "edit,name,expected",
    [
        # The argument named for rows multiplies the column index: bound by value, not by position.
        (
            _X.replace("(t + j)[:, None] * sxr + c[None, :] * sxc", "c[None, :] * sxr + (t + j)[:, None] * sxc"),
            "X",
            ("sxb", "sxh", "sxc", "sxr"),
        ),
        (_X.replace("b * sxb + h * sxh", "h * sxb + b * sxh"), "X", ("sxh", "sxb", "sxr", "sxc")),
        (_Y.replace("b * syb + h * syh", "h * syh + b * syb"), "Y", ("syb", "syh", "syr", "syc")),
        (_Y.replace("r[:, None] * syr", "(r * syr)[:, None]"), "Y", ("syb", "syh", "syr", "syc")),
    ],
    ids=["rows_by_column_name", "batch_head_names_swapped", "term_order", "multiply_then_lift"],
)
def test_witness_binds_an_equivalent_spelling_by_value(edit, name, expected):
    from triton_msl.codegen._address_witness import prove_template_addresses

    old = _X if name == "X" else _Y
    graph = _tile_graph((old, edit))
    roles, loop, index = _tile_roles(graph)
    assert prove_template_addresses(graph, roles, loop)[name] == [index[n] for n in expected]


_HEAD = "    h = bh % NH\n"
_HEAD_UNSIGNED = "    h = bh.to(tl.uint32) % NH.to(tl.uint32)\n"
_ZEXT = ".to(tl.uint32).to(tl.int64)"


@pytest.mark.parametrize(
    "edits",
    [
        # The remainder spelled by its definition: program_id - (program_id // NH) * NH.
        [(_HEAD, "    h = bh - b * NH\n")],
        [(_HEAD, "    h = bh - NH * (bh // NH)\n")],
        # A zero-extension is followed as the identity: where it is not (a negative operand),
        # the source address is a multiple of 2**32 elements away, outside any buffer a 32-bit
        # template addresses.
        [(_X, _X.replace("(t + j)", "(t + j)" + _ZEXT))],
        [(_X, _X.replace("c[None, :]", "c" + _ZEXT + "[None, :]"))],
        [(_Y, _Y.replace("r[:, None]", "r" + _ZEXT + "[:, None]"))],
        [(_X, _X.replace("(t + j)", "(((t + j) - N)" + _ZEXT + " + N)"))],
        # The unsigned batch decode: the head times its stride is zero-extended.
        [("    b = bh // NH\n", "    b = bh.to(tl.uint32) // NH.to(tl.uint32)\n"), (_HEAD, _HEAD_UNSIGNED)],
    ],
    ids=[
        "head_by_subtraction",
        "head_by_subtraction_inline",
        "zero_extended_key_index",
        "zero_extended_column_index",
        "zero_extended_row_index",
        "zero_extended_difference",
        "unsigned_batch_decode",
    ],
)
def test_witness_proves_an_equivalent_coordinate(edits):
    from triton_msl.codegen._address_witness import prove_template_addresses

    graph = _tile_graph(*edits)
    roles, loop, index = _tile_roles(graph)
    assert prove_template_addresses(graph, roles, loop) == {
        "X": [index[n] for n in ("sxb", "sxh", "sxr", "sxc")],
        "Y": [index[n] for n in ("syb", "syh", "syr", "syc")],
    }


@pytest.mark.parametrize(
    "edits,message",
    [
        ([(_X, _X.replace("(t + j)", "(t + j + 1)"))], "X: the source load address is not the template's"),
        (
            [(_X, _X.replace("(t + j)", "(t + j * 2)"))],
            "X: the source address has no stride argument where the template needs its slot-2 stride",
        ),
        ([(_X, _X.replace("c[None, :]", "(c + 1)[None, :]"))], "X: the source load address is not the template's"),
        ([(_X, _X.replace("(t + j)", "j"))], "X: the source load address is not the template's"),
        (
            [(_X, _X.replace("b * sxb + h * sxh", "(b + 1) * sxb + h * sxh"))],
            "X: the source load address is not the template's",
        ),
        (
            [(_X, _X.replace("b * sxb + h * sxh", "h * sxh"))],
            "X: the source address has no stride argument where the template needs its slot-0 stride",
        ),
        ([(_Y, _Y.replace("r[:, None]", "(r + 1)[:, None]"))], "Y: the source store address is not the template's"),
        (
            [(_Y, _Y.replace("b * syb", "bh * syb"))],
            "Y: the source address has no stride argument where the template needs its slot-0 stride",
        ),
        ([("range(0, N, B)", "range(0, N, 2 * B)")], "a source loop that is not range(0, sequence length, 4)"),
        ([("range(0, N, B)", "range(B, N, B)")], "a source loop that is not range(0, sequence length, 4)"),
        ([("range(0, N, B)", "range(0, NH, B)")], "a source loop that is not range(0, sequence length, 4)"),
        ([(_X, _X.replace("(t + j)", "tl.minimum(t + j, N)"))], "arith.minsi in an address"),
        (
            [(_X, _X.replace("(t + j)", "((t + j) // 2)"))],
            "a division in an address that is not a program id over a scalar",
        ),
        # Which of the two refusals names it depends on the two arguments' sampled values.
        ([("    b = bh // NH\n", "    b = bh // N\n")], "X: the source "),
        ([(_Y, "if pid == 0:\n        " + _Y)], "a conditional load or store"),
        ([(_Y, _Y + "\n    tl.store(Y + syc, 1.0)")], "Y: an access that is not one (4, 2) tile"),
        (
            [(_HEAD, "    h = bh - b * N\n")],
            "X: the source address has no stride argument where the template needs its slot-0 stride",
        ),
        (
            [(_HEAD, "    h = bh - (bh // N) * NH\n")],
            "X: the source address has no stride argument where the template needs its slot-0 stride",
        ),
        ([(_HEAD, "    h = bh - b * NH + 1\n")], "X: the source load address is not the template's"),
        # Below 32 bits a wrap is not a multiple of 2**32 elements away.
        (
            [(_X, _X.replace("(t + j)", "(t + j.to(tl.int8))"))],
            "an integer narrower than 32 bits in an address",
        ),
        (
            [(_X, _X.replace("(t + j)", "(t + j.to(tl.uint16))"))],
            "an integer narrower than 32 bits in an address",
        ),
    ],
    ids=[
        "row_plus_1",
        "row_scaled",
        "depth_plus_1",
        "loop_variable_dropped",
        "batch_plus_1",
        "batch_dropped",
        "store_row_plus_1",
        "batch_from_the_flat_program_id",
        "loop_step_doubled",
        "loop_start_moved",
        "loop_bound_other_argument",
        "minimum_in_the_address",
        "index_divided",
        "batch_divisor_other_argument",
        "conditional_store",
        "second_store_shape",
        "head_subtraction_other_divisor",
        "head_subtraction_other_quotient",
        "head_subtraction_plus_1",
        "index_through_8_bits",
        "index_through_16_bits",
    ],
)
def test_witness_refuses_a_different_address(edits, message):
    from triton_msl.codegen._address_witness import WitnessRefusal, prove_template_addresses

    graph = _tile_graph(*edits)
    roles, loop, _index = _tile_roles(graph)
    with pytest.raises(WitnessRefusal) as info:
        prove_template_addresses(graph, roles, loop)
    assert message in str(info.value)


def test_witness_refuses_accesses_and_loops_the_template_does_not_replay():
    from triton_msl.codegen._address_witness import WitnessRefusal, prove_template_addresses

    graph = _tile_graph()
    roles, loop, index = _tile_roles(graph)
    with pytest.raises(WitnessRefusal, match="a load or store on a pointer the template does not replay"):
        prove_template_addresses(graph, roles[1:], loop)
    with pytest.raises(WitnessRefusal, match="1 source loops where the template runs none"):
        prove_template_addresses(graph, roles, None)
    with pytest.raises(WitnessRefusal, match="two template roles on one pointer argument"):
        prove_template_addresses(graph, roles + [dict(roles[0], name="X2")], loop)
    with pytest.raises(WitnessRefusal, match="Y: a store where the template performs a load"):
        prove_template_addresses(graph, [roles[0], dict(roles[1], kind="load")], loop)
    with pytest.raises(WitnessRefusal, match="X: a tile axis shorter than two elements"):
        prove_template_addresses(graph, [dict(roles[0], shape=(4, 1)), roles[1]], loop)
    two_loops = _tile_graph(("    tl.store(", "    for u in range(0, N, B):\n        acc += 1.0\n    tl.store("))
    roles, loop, _ = _tile_roles(two_loops)
    with pytest.raises(WitnessRefusal, match="2 source loops where the template runs exactly one"):
        prove_template_addresses(two_loops, roles, loop)


def test_witness_reads_an_index_table_only_at_the_template_entries():
    from triton_msl.codegen._address_witness import WitnessRefusal, prove_template_addresses

    @triton.jit
    def _packed(X, CU, Y, sxt, sxc, syt, syc, B: tl.constexpr, D: tl.constexpr):
        b = tl.program_id(1)
        start = tl.load(CU + b)
        r = tl.program_id(0) * B + tl.arange(0, B)
        c = tl.arange(0, D)
        x = tl.load(X + (start + r)[:, None] * sxt + c[None, :] * sxc)
        tl.store(Y + (start + r)[:, None] * syt + c[None, :] * syc, x)

    def spec(graph):
        index = {a.name: a.index for a in graph.args}

        def role(name, ptr, kind, p):
            return {
                "name": name,
                "arg": index[ptr],
                "kind": kind,
                "strides": [index[p + "t"], index[p + "c"]],
                "shape": (4, 2),
                "scalars": [(0, lambda env: ("load", index["CU"], env.atom(("pid", 1))))],
                "axes": [
                    (0, 0, lambda env: [env.atom(("pid", 0)) * 4 + i for i in range(4)]),
                    (1, 1, lambda env: [0, 1]),
                ],
            }

        table = {
            "name": "cu",
            "arg": index["CU"],
            "table": lambda env: (env.atom(("pid", 1)), env.atom(("pid", 1)) + 1),
        }
        return [role("X", "X", "load", "sx"), role("Y", "Y", "store", "sy"), table]

    pointers = {"X": "*fp32", "CU": "*i32", "Y": "*fp32"}
    graph = _graph(_packed, _TILE_CONST, pointers)
    assert set(prove_template_addresses(graph, spec(graph), None)) == {"X", "Y"}
    # Without the table role, the loaded start is an index the witness cannot evaluate.
    with pytest.raises(
        WitnessRefusal, match="a loaded value in an address that is not one unmasked read of an index table"
    ):
        prove_template_addresses(graph, spec(graph)[:2], None)
    for old, new, message in [
        ("tl.load(CU + b)", "tl.load(CU + b + 2)", "cu: a read of an entry the template does not read"),
        (
            "(start + r)[:, None] * syt",
            "(start + 1 + r)[:, None] * syt",
            "Y: the source store address is not the template's",
        ),
    ]:
        fn = triton.JITFunction(_packed.fn)
        fn._unsafe_update_src(fn.src.replace(old, new))
        graph = _graph(fn, _TILE_CONST, pointers)
        with pytest.raises(WitnessRefusal, match=message):
            prove_template_addresses(graph, spec(graph), None)


# ---------------------------------------------------------------------------------------
# GPU helpers: float64 references evaluated from each variant's own address spelling
# ---------------------------------------------------------------------------------------


@pytest.fixture
def fa_hits(monkeypatch):
    """Record whether the FlashAttention replacement dispatch ran (the driver imports it per call)."""
    import triton_msl.autotuning._fa_dispatch as fa_dispatch

    hits, real = [], fa_dispatch.dispatch_flash_attention

    def spy(*args, **kwargs):
        result = real(*args, **kwargs)
        hits.append(bool(result))
        return result

    monkeypatch.setattr(fa_dispatch, "dispatch_flash_attention", spy)
    return hits


def _gather(storage, offsets, valid=None):
    """storage.flatten()[offsets] in float64; offsets outside ``valid`` read 0 (a masked load)."""
    flat = storage.reshape(-1).double()
    if valid is None:
        return flat[offsets]
    valid = valid.expand(offsets.shape)
    return torch.where(valid, flat[torch.where(valid, offsets, 0)], 0.0)


def _place(expected, offsets, values, keep):
    """Write ``values`` at ``offsets`` where ``keep``: the cells the source stores, each once."""
    keep = keep.expand(offsets.shape)
    cells = offsets[keep]
    assert cells.numel() == cells.unique().numel(), "a variant whose programs store one cell twice"
    expected.view(-1)[cells] = values.expand(offsets.shape)[keep]


def _assert_poisoned_match(got, expected, tol):
    got, expected = got.cpu().double(), expected.double()
    written = expected != POISON
    assert torch.equal(got != POISON, written), f"{int(((got != POISON) != written).sum())} cells differ in domain"
    torch.testing.assert_close(got[written], expected[written], atol=tol, rtol=tol)


def _refused_or_computed(launch, expected_outputs, fragment, hits, tol=2e-3):
    """Run a variant: a refusal must carry ``fragment`` (or one of a tuple of them); a run must
    equal the source reference. ``fragment=None`` marks a program that must compute."""
    try:
        outputs = launch()
    except MetalNonRecoverableError as exc:
        assert fragment is not None, f"refused a variant that must compute: {exc}"
        fragments = (fragment,) if isinstance(fragment, str) else fragment
        assert any(f in str(exc) for f in fragments), str(exc)
        assert not any(hits)
        return "refused"
    for got, expected in zip(outputs, expected_outputs()):
        _assert_poisoned_match(got, expected, tol)
    return "computed"


# ---------------------------------------------------------------------------------------
# GPU: dense FlashAttention (tests/test_flash_attention.py::_flash_attn_fwd)
# ---------------------------------------------------------------------------------------

_DQ = "offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk"
_DK = "+ (start_n + offs_n)[:, None] * stride_kn\n            + offs_d[None, :] * stride_kk"
_DV = "+ (start_n + offs_n)[:, None] * stride_vn\n            + offs_d[None, :] * stride_vk"
_DO = "offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok"
_DSTORE = "mask=offs_m[:, None] < N_CTX)\n"


def _dense_addr(s, p1, heads, x, d):
    return (p1 // heads) * s[0] + (p1 % heads) * s[1] + x * s[2] + d * s[3]


def _dense_case(
    edits=(), fragment=None, q=None, k=None, v=None, o=None, store=None, n_ctx=80, q_order=None, dims=(64, 128)
):
    return dict(
        edits=edits, fragment=fragment, q=q, k=k, v=v, o=o, store=store, n_ctx=n_ctx, q_order=q_order, dims=dims
    )


_FA_TAIL = "FlashAttention Tail value path is not exactly reproduced"
_FA_CLIP = "with a non-tile-boundary output store mask is not supported"
# The generic path these run on refuses head_dim 128 outright.
_FA_GENERIC = "FlashAttention head_dim > 64 (a dot tile dimension is 128) is not supported"
DENSE_CASES = {
    "canonical": _dense_case(),
    # D1a-D1c: the sequence length clips COLUMNS; the template writes every head-dim column.
    "col_and_row_nctx": _dense_case(
        [(_DSTORE, "mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < N_CTX))\n")],
        _FA_CLIP,
        store=lambda m, d, n: (m < n) & (d < n),
        n_ctx=32,
    ),
    "col_only_nctx": _dense_case(
        [(_DSTORE, "mask=offs_d[None, :] < N_CTX)\n")], _FA_CLIP, store=lambda m, d, n: d < n, n_ctx=32
    ),
    "lifted_col_nctx": _dense_case(
        [(_DSTORE, "mask=(offs_m < N_CTX)[:, None] & (offs_d < N_CTX)[None, :])\n")],
        _FA_CLIP,
        store=lambda m, d, n: (m < n) & (d < n),
        n_ctx=32,
    ),
    "q_row_plus_1": _dense_case(
        [(_DQ, _DQ.replace("offs_m[", "(offs_m + 1)["))],
        _FA_TAIL,
        q=lambda s, p, hh, m, d: _dense_addr(s, p, hh, m + 1, d),
    ),
    "q_depth_plus_1": _dense_case(
        [(_DQ, _DQ.replace("offs_d[", "(offs_d + 1)["))],
        _FA_TAIL,
        q=lambda s, p, hh, m, d: _dense_addr(s, p, hh, m, d + 1),
    ),
    "k_loop_variable_dropped": _dense_case(
        [(_DK, _DK.replace("(start_n + offs_n)[", "offs_n["))],
        _FA_TAIL,
        k=lambda s, p, hh, n, d: _dense_addr(s, p, hh, n % 32, d),
    ),
    "v_row_plus_1": _dense_case(
        [(_DV, _DV.replace("(start_n + offs_n)[", "(start_n + offs_n + 1)["))],
        _FA_TAIL,
        v=lambda s, p, hh, n, d: _dense_addr(s, p, hh, n + 1, d),
    ),
    # Batch and head offsets on the wrong strides, and a batch offset shifted: not the
    # template's coordinates (these run on the generic path and must still be the source's).
    "q_batch_head_swapped": _dense_case(
        [("Q + off_z * stride_qz + off_h * stride_qh", "Q + off_h * stride_qz + off_z * stride_qh")],
        _FA_GENERIC,
        q=lambda s, p, hh, m, d: (p % hh) * s[0] + (p // hh) * s[1] + m * s[2] + d * s[3],
    ),
    "o_batch_plus_1": _dense_case(
        [("Out + off_z * stride_oz", "Out + (off_z + 1) * stride_oz")],
        _FA_GENERIC,
        o=lambda s, p, hh, m, d: _dense_addr(s, p, hh, m, d) + s[0],
    ),
    # Equivalent programs: must compute. The reordered terms run on the generic path at
    # head_dim 64; the head-dim-tiled route refuses them at 128 ("row stride resolved to 1"),
    # an over-refusal outside the address witness (not exercised here).
    "eq_q_term_order": _dense_case([(_DQ, "offs_d[None, :] * stride_qk + offs_m[:, None] * stride_qm")], dims=(64,)),
    "eq_o_term_order": _dense_case([(_DO, "offs_d[None, :] * stride_ok + offs_m[:, None] * stride_om")], dims=(64,)),
    "eq_q_strides_declared_in_another_order": _dense_case(
        [("    stride_qm,\n    stride_qk,\n", "    stride_qk,\n    stride_qm,\n")], q_order=(0, 1, 3, 2)
    ),
}


@requires_mps
@pytest.mark.parametrize(
    "name,head_dim,causal", [(n, d, d == 128) for n in sorted(DENSE_CASES) for d in DENSE_CASES[n]["dims"]]
)
def test_dense_variant_refuses_or_computes_the_source(name, head_dim, causal, fa_hits):
    from tests.test_flash_attention import _flash_attn_fwd

    case = DENSE_CASES[name]
    fn = _variant(_flash_attn_fwd, *case["edits"])
    z, heads, n_ctx = 2, 2, case["n_ctx"]
    tiles = triton.cdiv(n_ctx, 32) + (1 if n_ctx == 32 else 0)  # a column clip at 32 runs a second tile
    rows = tiles * 32
    gen = torch.Generator().manual_seed(1201 + head_dim)
    q, k, v = [torch.randn(z + 1, heads + 1, 2 * rows + 32, head_dim + 2, generator=gen) for _ in range(3)]
    out = torch.full((z + 2, heads + 1, rows + 32, head_dim + 2), POISON)
    views = [x[:z, :heads, :rows, :head_dim] for x in (q, k, v)]
    sq, sk, sv, so = q.stride(), k.stride(), v.stride(), out.stride()

    def launch():
        dq, dk, dv = [x.to("mps") for x in (q, k, v)]
        dout = out.to("mps")
        qv, kv, vv = [x[:z, :heads, :rows, :head_dim] for x in (dq, dk, dv)]
        ov = dout[:z, :heads, :rows, :head_dim]
        q_strides = qv.stride() if case["q_order"] is None else [qv.stride()[i] for i in case["q_order"]]
        fn[(tiles, z * heads)](
            qv, kv, vv, ov, *q_strides, *kv.stride(), *vv.stride(), *ov.stride(), z, heads, n_ctx,
            BLOCK_M=32, BLOCK_N=32, HEAD_DIM=head_dim, IS_CAUSAL=causal,
        )  # fmt: skip
        torch.mps.synchronize()
        return [dout]

    def expected():
        exp = torch.full(out.shape, POISON, dtype=torch.float64)
        d = torch.arange(head_dim)[None, :]
        for p in range(z * heads):
            for pid in range(tiles):
                m = torch.arange(pid * 32, pid * 32 + 32)[:, None]
                qe = _gather(q, (case["q"] or _dense_addr)(sq, p, heads, m, d), m < n_ctx) * head_dim**-0.5
                hi = min((pid + 1) * 32, n_ctx) if causal else n_ctx
                n = torch.arange(triton.cdiv(hi, 32) * 32)[:, None]
                ke = _gather(k, (case["k"] or _dense_addr)(sk, p, heads, n, d), n < n_ctx)
                ve = _gather(v, (case["v"] or _dense_addr)(sv, p, heads, n, d), n < n_ctx)
                s = qe @ ke.T
                if causal:
                    s = s.masked_fill(m < n.T, float("-inf"))
                values = torch.softmax(s, -1) @ ve
                keep = (case["store"] or (lambda m, d, n: m < n))(m, d, n_ctx)
                _place(exp, (case["o"] or _dense_addr)(so, p, heads, m, d), values, keep)
        return [exp]

    assert views[0].shape == (z, heads, rows, head_dim)
    verdict = _refused_or_computed(launch, expected, case["fragment"], fa_hits)
    if case["fragment"] is None:
        assert verdict == "computed"


# ---------------------------------------------------------------------------------------
# GPU: biased / triangle FlashAttention forward, 2-D grid (tests/test_fa_biased_routing.py)
# ---------------------------------------------------------------------------------------

_BQ = "q_ptrs = q_ptr + z * q_sz + h * q_sh + m_idxs[:, None] * q_sm + d_idxs[None, :] * q_sk"
_BK = "kt_ptrs = k_ptr + z * k_sz + h * k_sh + d_idxs[:, None] * k_sk + n_idxs[None, :] * k_sn"
_BV = "v_ptrs = v_ptr + z * v_sz + h * v_sh + n_idxs[:, None] * v_sn + d_idxs[None, :] * v_sk"
_BB = "b_ptrs = b_ptr + z * b_sz + h * b_sh + m_idxs[:, None] * b_sm + n_idxs[None, :] * b_sn"
_BO = "o_ptrs = o_ptr + z * o_sz + h * o_sh + m_idxs[:, None] * o_sm + d_idxs[None, :] * o_sk"
_BL = "lse_ptrs = lse_ptr + z * lse_sz + h * lse_sh + m_idxs * lse_sm"
_BOS = "tl.store(o_ptrs, normalize.to(q_block.type.element_ty), mask=mask_m[:, None])"
_BLS = "tl.store(lse_ptrs, lse, mask=mask_m)"


def _a4(s, a, b, x, y):
    return a * s[0] + b * s[1] + x * s[2] + y * s[3]


def _a3(s, a, b, x):
    return a * s[0] + b * s[1] + x * s[2]


# Address of each role in (strides, batch, head, tile index, tile index); the mask role takes
# the head count too (the 3-D route decodes its batch as program id 2 // H).
BIASED_CANON = dict(
    q=_a4, k=_a4, v=_a4, b=_a4, o=_a4, lse=_a3, mask=lambda s, a, b, n, heads: _a3(s, a, b, n), keys=None
)


def _biased_case(edits=(), fragment=None, n=64, dim=32, q_order=None, **overrides):
    return dict(edits=edits, fragment=fragment, n=n, dim=dim, q_order=q_order, **{**BIASED_CANON, **overrides})


def _biased_expected(t, case, programs, decode, heads, rows):
    """Float64 semantics of the biased forward source for every program; Out and Lse cells."""
    n_ctx, dim = case["n"], case["dim"]
    s = {name: x.stride() for name, x in t.items()}
    o_exp = torch.full(t["o"].shape, POISON, dtype=torch.float64)
    l_exp = torch.full(t["lse"].shape, POISON, dtype=torch.float64)
    m = torch.arange(rows)[:, None]
    d = torch.arange(dim)[None, :]
    n = (torch.arange(n_ctx) if case["keys"] is None else case["keys"](n_ctx))[:, None]
    for p1, p2 in programs:
        a, b = decode(p1, p2)
        qe = _gather(t["q"], case["q"](s["q"], a, b, m, d)) * dim**-0.5
        ke = _gather(t["k"], case["k"](s["k"], a, b, n, d))
        ve = _gather(t["v"], case["v"](s["v"], a, b, n, d))
        be = _gather(t["b"], case["b"](s["b"], a, b, m, n.T))
        me = _gather(t["mask"], case["mask"](s["mask"], a, b, n.T, heads))
        raw = (qe @ ke.T + be).masked_fill(me != 0, float("-inf"))
        _place(
            o_exp,
            case["o"](s["o"], a, b, m, d),
            torch.softmax(raw, -1) @ ve,
            case.get("omask", _rows_below)(m, d, n_ctx),
        )
        lse_keep = case.get("lmask", lambda m, n: m < n)(m, n_ctx)
        _place(l_exp, case["lse"](s["lse"], a, b, m), torch.logsumexp(raw, -1, keepdim=True), lse_keep)
    return [o_exp, l_exp]


def _rows_below(m, d, n):
    return m < n


def _biased_inputs(case, outer, seed):
    """Every tensor larger than the logical one in every axis, so a shifted index stays in storage."""
    gen = torch.Generator().manual_seed(seed)
    n, dim = case["n"], case["dim"]
    big = 2 * n + 64
    t = {name: torch.randn(*outer, big, dim + 2, generator=gen) for name in ("q", "k", "v")}
    t["b"] = torch.randn(*outer, big, big + 1, generator=gen)
    t["mask"] = (torch.rand(*outer, big, generator=gen) < 0.2).to(torch.uint8)
    t["o"] = torch.full((outer[0] + 1, outer[1] + 1, big, dim + 2), POISON)
    t["lse"] = torch.full((outer[0] + 1, outer[1] + 1, big), POISON)
    return t


def _biased_args(dev, case, lead, rows, bias_lead=None):
    """Kernel arguments: Out, Lse, Q, K, V, Bias, Mask views and their strides."""
    dim = case["dim"]
    views = {}
    for name in ("o", "q", "k", "v"):
        views[name] = dev[name][lead + (slice(0, rows), slice(0, dim))]
    views["lse"] = dev["lse"][lead + (slice(0, rows),)]
    views["b"] = dev["b"][(lead if bias_lead is None else bias_lead) + (slice(0, rows), slice(0, rows))]
    views["mask"] = dev["mask"][lead + (slice(0, rows),)]
    args = []
    for name in ("o", "lse", "q", "k", "v", "b", "mask"):
        strides = list(views[name].stride())
        if name == "q" and case["q_order"] is not None:
            strides = [strides[i] for i in case["q_order"]]
        args += [views[name], *strides]
    return args


_BIASED_ADDR = "Biased/triangle FlashAttention {} address is not exactly the template's"
BIASED_CASES = {
    "canonical": _biased_case(),
    "q_row_plus_1": _biased_case(
        [(_BQ, _BQ.replace("m_idxs[", "(m_idxs + 1)["))],
        _BIASED_ADDR.format("Q"),
        q=lambda s, a, b, m, d: _a4(s, a, b, m + 1, d),
    ),
    "q_row_scaled": _biased_case(
        [(_BQ, _BQ.replace("m_idxs[", "(m_idxs * 2)["))],
        _BIASED_ADDR.format("Q"),
        q=lambda s, a, b, m, d: _a4(s, a, b, 2 * m, d),
    ),
    "q_depth_plus_1": _biased_case(
        [(_BQ, _BQ.replace("d_idxs[", "(d_idxs + 1)["))],
        _BIASED_ADDR.format("Q"),
        q=lambda s, a, b, m, d: _a4(s, a, b, m, d + 1),
    ),
    "k_loop_advance_doubled": _biased_case(
        [("kt_ptrs += BLOCK_N * k_sn", "kt_ptrs += 2 * BLOCK_N * k_sn")],
        _BIASED_ADDR.format("K"),
        k=lambda s, a, b, n, d: _a4(s, a, b, n + 32 * (n // 32), d),
    ),
    "v_loop_variable_dropped": _biased_case(
        [("        v_ptrs += BLOCK_N * v_sn\n", "")],
        _BIASED_ADDR.format("V"),
        v=lambda s, a, b, n, d: _a4(s, a, b, n % 32, d),
    ),
    "loop_step_doubled": _biased_case(
        [("tl.range(0, N, BLOCK_N)", "tl.range(0, N, 2 * BLOCK_N)")],
        _BIASED_ADDR.format("K/V"),
        keys=lambda n: torch.arange(32),
    ),
    "bias_batch_plus_1": _biased_case(
        [(_BB, _BB.replace("z * b_sz", "(z + 1) * b_sz"))],
        _BIASED_ADDR.format("Bias"),
        b=lambda s, a, b, m, n: _a4(s, a + 1, b, m, n),
    ),
    "mask_head_plus_1": _biased_case(
        [("h * m_sh", "(h + 1) * m_sh")],
        _BIASED_ADDR.format("Mask"),
        mask=lambda s, a, b, n, heads: _a3(s, a, b + 1, n),
    ),
    "out_batch_plus_1": _biased_case(
        [(_BO, _BO.replace("z * o_sz", "(z + 1) * o_sz"))],
        _BIASED_ADDR.format("Out"),
        o=lambda s, a, b, m, d: _a4(s, a + 1, b, m, d),
    ),
    "lse_head_plus_1": _biased_case(
        [(_BL, _BL.replace("h * lse_sh", "(h + 1) * lse_sh"))],
        _BIASED_ADDR.format("Lse"),
        lse=lambda s, a, b, m: _a3(s, a, b + 1, m),
    ),
    # D1d: the sequence length clips head-dim columns (N = 32 < DIM = 64).
    "out_columns_clipped_by_n": _biased_case(
        [(_BOS, _BOS.replace("mask=mask_m[:, None])", "mask=mask_m[:, None] & (d_idxs[None, :] < N))"))],
        "with a non-tile-boundary output store mask is not supported",
        n=32,
        dim=64,
        omask=lambda m, d, n: (m < n) & (d < n),
    ),
    # D2: Out rows shifted in the address AND the mask; the template stores at the bare row.
    "out_row_shift_with_its_mask": _biased_case(
        [
            (_BO, _BO.replace("m_idxs[", "(m_idxs + 1)[")),
            (_BOS, _BOS.replace("mask_m[:, None]", "(m_idxs + 1)[:, None] < N")),
        ],
        "FlashAttention Out value path is not exactly reproduced",
        o=lambda s, a, b, m, d: _a4(s, a, b, m + 1, d),
        omask=lambda m, d, n: m + 1 < n,
    ),
    # D3a / D3b: Lse rows shifted under the canonical mask (lse_sm == 1 folds into the index).
    "lse_row_shift_canonical_mask": _biased_case(
        [(_BL, _BL.replace("m_idxs * lse_sm", "(m_idxs + 1) * lse_sm"))],
        _BIASED_ADDR.format("Lse"),
        lse=lambda s, a, b, m: _a3(s, a, b, m + 1),
    ),
    "lse_plus_1_unit_stride": _biased_case(
        [(_BL, _BL.replace("m_idxs * lse_sm", "(m_idxs + 1)"))],
        _BIASED_ADDR.format("Lse"),
        lse=lambda s, a, b, m: a * s[0] + b * s[1] + m + 1,
    ),
    # Different programs the template replays exactly: must compute.
    "q_batch_head_swapped": _biased_case(
        [(_BQ, _BQ.replace("z * q_sz + h * q_sh", "h * q_sz + z * q_sh"))], q=lambda s, a, b, m, d: _a4(s, b, a, m, d)
    ),
    "bias_head_dropped": _biased_case(
        [(_BB, _BB.replace("z * b_sz + h * b_sh", "z * b_sz"))], b=lambda s, a, b, m, n: _a4(s, a, 0, m, n)
    ),
    # Equivalent spellings of the canonical program: must compute.
    "eq_q_term_order": _biased_case(
        [
            (
                _BQ,
                _BQ.replace(
                    "m_idxs[:, None] * q_sm + d_idxs[None, :] * q_sk", "d_idxs[None, :] * q_sk + m_idxs[:, None] * q_sm"
                ),
            )
        ]
    ),
    "eq_k_term_order": _biased_case(
        [
            (
                _BK,
                _BK.replace(
                    "d_idxs[:, None] * k_sk + n_idxs[None, :] * k_sn", "n_idxs[None, :] * k_sn + d_idxs[:, None] * k_sk"
                ),
            )
        ]
    ),
    "eq_o_term_order": _biased_case(
        [
            (
                _BO,
                _BO.replace(
                    "m_idxs[:, None] * o_sm + d_idxs[None, :] * o_sk", "d_idxs[None, :] * o_sk + m_idxs[:, None] * o_sm"
                ),
            )
        ]
    ),
    "eq_q_multiply_then_lift": _biased_case([(_BQ, _BQ.replace("m_idxs[:, None] * q_sm", "(m_idxs * q_sm)[:, None]"))]),
    "eq_q_strides_declared_in_another_order": _biased_case(
        [("    q_sm,\n    q_sk,\n", "    q_sk,\n    q_sm,\n")], q_order=(0, 1, 3, 2)
    ),
    # The head index by the remainder's definition: must compute on the template.
    "eq_head_by_subtraction": _biased_case([("h = pid_zh % H", "h = pid_zh - z * H")]),
}


@requires_mps
@pytest.mark.parametrize("name", sorted(BIASED_CASES))
def test_biased_variant_refuses_or_computes_the_source(name, fa_hits):
    from tests.test_fa_biased_routing import _biased_fa

    case = BIASED_CASES[name]
    fn = _variant(_biased_fa, *case["edits"])
    z, heads, tiles = 2, 2, triton.cdiv(case["n"], 32)
    t = _biased_inputs(case, (z + 1, heads + 1), seed=1301)

    def launch():
        dev = {k: x.to("mps") for k, x in t.items()}
        args = _biased_args(dev, case, (slice(0, z), slice(0, heads)), tiles * 32)
        fn[(tiles, z * heads)](*args, case["dim"] ** -0.5, -1e9, z, heads, case["n"], case["dim"], 32, 32)
        torch.mps.synchronize()
        return [dev["o"], dev["lse"]]

    def expected():
        programs = [(p, 0) for p in range(z * heads)]
        return _biased_expected(t, case, programs, lambda p1, p2: (p1 // heads, p1 % heads), heads, tiles * 32)

    verdict = _refused_or_computed(launch, expected, case["fragment"], fa_hits)
    if verdict == "computed" and case["fragment"] is None:
        assert fa_hits == [True]


# ---------------------------------------------------------------------------------------
# GPU: biased forward on the 3-D triangle grid (program ids j-tile, i, h; mask batch h // H)
# ---------------------------------------------------------------------------------------

_TQ = "q_ptrs = q_ptr + pid_h * q_sh + pid_i * q_si + j_idxs[:, None] * q_sn + d_idxs[None, :] * q_sd"
_TM = "mask_ptrs = mask_ptr + mask_start_h * mask_sh + pid_i * mask_si + k_idxs * mask_sn"
_TL = "lse_ptrs = lse_ptr + pid_h * lse_sh + pid_i * lse_si + j_idxs * lse_sn"
_TB = "b_ptrs = b_ptr + pid_h * b_sh + j_idxs[:, None] * b_sn + k_idxs[None, :] * b_sm"


def _tri_bias(s, h, i, m, n):
    return h * s[0] + m * s[1] + n * s[2]


def _tri_mask(s, h, i, n, heads):
    return (h // heads) * s[0] + i * s[1] + n * s[2]


TRI_CASES = {
    "canonical": _biased_case(b=_tri_bias, mask=_tri_mask),
    "q_instance_plus_1": _biased_case(
        [(_TQ, _TQ.replace("pid_i * q_si", "(pid_i + 1) * q_si"))],
        _BIASED_ADDR.format("Q"),
        b=_tri_bias,
        mask=_tri_mask,
        q=lambda s, h, i, m, d: _a4(s, h, i + 1, m, d),
    ),
    "lse_head_plus_1": _biased_case(
        [(_TL, _TL.replace("pid_h * lse_sh", "(pid_h + 1) * lse_sh"))],
        _BIASED_ADDR.format("Lse"),
        b=_tri_bias,
        mask=_tri_mask,
        lse=lambda s, h, i, m: _a3(s, h + 1, i, m),
    ),
    "bias_head_plus_1": _biased_case(
        [(_TB, _TB.replace("pid_h * b_sh", "(pid_h + 1) * b_sh"))],
        _BIASED_ADDR.format("Bias"),
        b=lambda s, h, i, m, n: _tri_bias(s, h + 1, i, m, n),
        mask=_tri_mask,
    ),
    "mask_batch_shifted": _biased_case(
        [("mask_start_h = pid_h // H", "mask_start_h = (pid_h + 1) // H")],
        _BIASED_ADDR.format("Mask"),
        b=_tri_bias,
        mask=lambda s, h, i, n, heads: _tri_mask(s, h + 1, i, n, heads),
    ),
    "mask_instance_dropped": _biased_case(
        [(_TM, _TM.replace(" + pid_i * mask_si", ""))],
        _BIASED_ADDR.format("Mask"),
        b=_tri_bias,
        mask=lambda s, h, i, n, heads: _tri_mask(s, h, 0, n, heads),
    ),
    "grid_axes_swapped": _biased_case(
        [
            (
                "    pid_i = tl.program_id(1)\n    pid_h = tl.program_id(2)\n",
                "    pid_i = tl.program_id(2)\n    pid_h = tl.program_id(1)\n",
            )
        ],
        "address is not exactly the template's",
        # Canonical (h, i) = (program id 2, program id 1); the source reads them the other way.
        q=lambda s, h, i, m, d: _a4(s, i, h, m, d),
        k=lambda s, h, i, n, d: _a4(s, i, h, n, d),
        v=lambda s, h, i, n, d: _a4(s, i, h, n, d),
        b=lambda s, h, i, m, n: _tri_bias(s, i, h, m, n),
        mask=lambda s, h, i, n, heads: _tri_mask(s, i, h, n, heads),
        o=lambda s, h, i, m, d: _a4(s, i, h, m, d),
        lse=lambda s, h, i, m: _a3(s, i, h, m),
    ),
    # Must compute: the template binds the batch strides by value / reads the mask at pid_h.
    "q_head_instance_swapped": _biased_case(
        [(_TQ, _TQ.replace("pid_h * q_sh + pid_i * q_si", "pid_i * q_sh + pid_h * q_si"))],
        b=_tri_bias,
        mask=_tri_mask,
        q=lambda s, h, i, m, d: _a4(s, i, h, m, d),
    ),
    "mask_batch_is_the_head": _biased_case(
        [("mask_start_h = pid_h // H", "mask_start_h = pid_h")],
        b=_tri_bias,
        mask=lambda s, h, i, n, heads: _tri_mask(s, h, i, n, 1),
    ),
    "eq_mask_term_order": _biased_case(
        [(_TM, _TM.replace("mask_start_h * mask_sh + pid_i * mask_si", "pid_i * mask_si + mask_start_h * mask_sh"))],
        b=_tri_bias,
        mask=_tri_mask,
    ),
}


@requires_mps
@pytest.mark.parametrize("name", sorted(TRI_CASES))
def test_triangle_variant_refuses_or_computes_the_source(name, fa_hits):
    from tests.test_fa_biased_routing import _biased_tri_fa

    case = TRI_CASES[name]
    fn = _variant(_biased_tri_fa, *case["edits"])
    hc, hh, inst, tiles = 4, 2, 3, triton.cdiv(case["n"], 32)
    # Every leading axis holds the larger of the head and instance counts, so a swap stays in storage.
    t = _biased_inputs(case, (hc + 1, hc + 1), seed=1302)
    t["b"] = t["b"][:, 0].contiguous()  # the bias is shared across the instance axis: [h, j, k]

    def launch():
        dev = {k: x.to("mps") for k, x in t.items()}
        args = _biased_args(dev, case, (slice(0, hc), slice(0, inst)), tiles * 32, bias_lead=(slice(0, hc),))
        fn[(tiles, inst, hc)](*args, case["dim"] ** -0.5, -1e9, case["n"], hh, case["dim"], 32, 32)
        torch.mps.synchronize()
        return [dev["o"], dev["lse"]]

    def expected():
        programs = [(i, h) for i in range(inst) for h in range(hc)]
        return _biased_expected(t, case, programs, lambda p1, p2: (p2, p1), hh, tiles * 32)

    verdict = _refused_or_computed(launch, expected, case["fragment"], fa_hits)
    if verdict == "computed" and case["fragment"] is None:
        assert fa_hits == [True]


# ---------------------------------------------------------------------------------------
# GPU: varlen FlashAttention (tests/test_varlen_fa_routing.py::_varlen_fwd)
# ---------------------------------------------------------------------------------------

_VQ = "(q_start + offs_m)[:, None] * stride_qt"
_VK = "(k_start + kn)[:, None] * stride_kt"
_VVD = "offs_d[None, :] * stride_vd"
_VOD = "offs_d[None, :] * stride_od"


def _packed(s, start, row, h, d):
    return (start + row) * s[0] + h * s[1] + d * s[2]


VARLEN_FRAGMENT = "varlen FlashAttention:"


def _varlen_case(edits=(), fragment=None, q_order=None, template=True, **overrides):
    return dict(
        edits=edits,
        fragment=fragment,
        q_order=q_order,
        template=template,
        **{**dict(q=_packed, k=_packed, v=_packed, o=_packed), **overrides},
    )


VARLEN_CASES = {
    "canonical": _varlen_case(),
    "q_row_plus_1": _varlen_case(
        [(_VQ, "(q_start + offs_m + 1)[:, None] * stride_qt")],
        VARLEN_FRAGMENT,
        q=lambda s, st, r, h, d: _packed(s, st, r + 1, h, d),
    ),
    "k_sequence_start_plus_1": _varlen_case(
        [(_VK, "(k_start + 1 + kn)[:, None] * stride_kt")],
        VARLEN_FRAGMENT,
        k=lambda s, st, r, h, d: _packed(s, st + 1, r, h, d),
    ),
    "k_loop_advance_doubled": _varlen_case(
        [(_VK, "(k_start + kn + start_n)[:, None] * stride_kt")],
        VARLEN_FRAGMENT,
        k=lambda s, st, r, h, d: _packed(s, st, r + 32 * (r // 32), h, d),
    ),
    "v_depth_plus_1": _varlen_case(
        [(_VVD, "(offs_d + 1)[None, :] * stride_vd")],
        VARLEN_FRAGMENT,
        v=lambda s, st, r, h, d: _packed(s, st, r, h, d + 1),
    ),
    "o_depth_plus_1": _varlen_case(
        [(_VOD, "(offs_d + 1)[None, :] * stride_od")],
        VARLEN_FRAGMENT,
        o=lambda s, st, r, h, d: _packed(s, st, r, h, d + 1),
    ),
    "q_head_is_the_batch": _varlen_case(
        [("off_h * stride_qh", "off_b * stride_qh")],
        VARLEN_FRAGMENT,
        q_head="batch",  # the reference reads Q at head = batch
    ),
    # Equivalent spellings: must compute (the distributed row term ran another program before).
    "eq_q_row_term_distributed": _varlen_case([(_VQ, "q_start * stride_qt + offs_m[:, None] * stride_qt")]),
    "eq_k_head_term_first": _varlen_case(
        [
            (
                "K + (k_start + kn)[:, None] * stride_kt + off_h * stride_kh",
                "K + off_h * stride_kh + (k_start + kn)[:, None] * stride_kt",
            )
        ]
    ),
    "eq_q_strides_declared_in_another_order": _varlen_case(
        [("    stride_qt,\n    stride_qh,\n", "    stride_qh,\n    stride_qt,\n")], q_order=(1, 0, 2)
    ),
    # The head index by the remainder's definition. The varlen detector reads the head count
    # from a remainder and declines this spelling before any address is proved, so the program
    # computes on the generic path: the source's result, without the template.
    "eq_head_by_subtraction": _varlen_case([("off_h = off_bh % H", "off_h = off_bh - off_b * H")], template=False),
}


@requires_mps
@pytest.mark.parametrize("name", sorted(VARLEN_CASES))
def test_varlen_variant_refuses_or_computes_the_source(name, fa_hits):
    from tests.test_varlen_fa_routing import _varlen_fwd

    case = VARLEN_CASES[name]
    fn = _variant(_varlen_fwd, *case["edits"])
    lens_q, lens_k, heads, dim = [37, 50, 21], [45, 29, 60], 2, 64
    maxs, scale = max(lens_q + lens_k), 1.0 / math.sqrt(dim)
    gen = torch.Generator().manual_seed(1401)
    cu_q = torch.tensor([0] + torch.tensor(lens_q).cumsum(0).tolist(), dtype=torch.int32)
    cu_k = torch.tensor([0] + torch.tensor(lens_k).cumsum(0).tolist(), dtype=torch.int32)
    tq, tk = int(cu_q[-1]), int(cu_k[-1])
    t = {
        "q": torch.randn(tq + 200, heads + 1, dim + 2, generator=gen),
        "k": torch.randn(tk + 200, heads + 1, dim + 2, generator=gen),
        "v": torch.randn(tk + 200, heads + 1, dim + 2, generator=gen),
        "o": torch.full((tq + 200, heads + 1, dim + 2), POISON),
    }

    def launch():
        dev = {n: x.to("mps") for n, x in t.items()}
        q, k, v, o = (dev[n][:, :heads, :dim] for n in ("q", "k", "v", "o"))
        q_strides = q.stride() if case["q_order"] is None else [q.stride()[i] for i in case["q_order"]]
        fn[(triton.cdiv(max(lens_q), 32), len(lens_q) * heads)](
            q, k, v, o, cu_q.to("mps"), cu_k.to("mps"), *q_strides, *k.stride(), *v.stride(), *o.stride(),
            heads, maxs, scale, 32, 32, dim,
        )  # fmt: skip
        torch.mps.synchronize()
        return [dev["o"]]

    def expected():
        s = {n: x.stride() for n, x in t.items()}
        exp = torch.full(t["o"].shape, POISON, dtype=torch.float64)
        d = torch.arange(dim)[None, :]
        kn = torch.arange(triton.cdiv(maxs, 32) * 32)[:, None]  # range(0, max_seqlen, 32) whole tiles
        for b in range(len(lens_q)):
            qs, ks = int(cu_q[b]), int(cu_k[b])
            for h in range(heads):
                qh = b if case.get("q_head") == "batch" else h
                for pid in range(triton.cdiv(max(lens_q), 32)):
                    m = torch.arange(pid * 32, pid * 32 + 32)[:, None]
                    qe = _gather(t["q"], case["q"](s["q"], qs, m, qh, d), m < lens_q[b]) * scale
                    ke = _gather(t["k"], case["k"](s["k"], ks, kn, h, d), kn < lens_k[b])
                    ve = _gather(t["v"], case["v"](s["v"], ks, kn, h, d), kn < lens_k[b])
                    sc = (qe @ ke.T).masked_fill(kn.T >= lens_k[b], float("-inf"))
                    _place(exp, case["o"](s["o"], qs, m, h, d), torch.softmax(sc, -1) @ ve, m < lens_q[b])
        return [exp]

    verdict = _refused_or_computed(launch, expected, case["fragment"], fa_hits, tol=1e-3)
    if verdict == "computed" and case["fragment"] is None:
        assert fa_hits == ([True] if case["template"] else [])


# ---------------------------------------------------------------------------------------
# GPU: MLA attention (tests/test_mla_value_paths.py::_mla_value_path MODE 0, fp16, BN 32)
# ---------------------------------------------------------------------------------------

_MQN = "qn_ptr = Qn + z * qnz + h * qnh + m[:, None] * qnm + dn[None, :] * qnk"
_MKN = "(s + n0)[:, None] * knn"
_MKR = "dr[None, :] * krk"
_MV = "(s + n0)[:, None] * vn"
_MO = "out_ptr = Out + z * oz + h * oh + m[:, None] * om + dv[None, :] * ok_"
MLA_FRAGMENT = "MLA attention:"
MLA_CANON = dict(qn=_a4, qr=_a4, kn=_a4, kr=_a4, v=_a4, o=_a4)


def _mla_case(edits=(), fragment=None, **overrides):
    return dict(edits=edits, fragment=fragment, **{**MLA_CANON, **overrides})


MLA_CASES = {
    "canonical": _mla_case(),
    "qn_batch_plus_1": _mla_case(
        [(_MQN, _MQN.replace("z * qnz", "(z + 1) * qnz"))],
        MLA_FRAGMENT,
        qn=lambda s, a, b, r, c: _a4(s, a + 1, b, r, c),
    ),
    "kn_row_plus_1": _mla_case(
        [(_MKN, "(s + n0 + 1)[:, None] * knn")], MLA_FRAGMENT, kn=lambda s, a, b, r, c: _a4(s, a, b, r + 1, c)
    ),
    "kr_depth_scaled": _mla_case(
        [(_MKR, "(dr * 2)[None, :] * krk")], MLA_FRAGMENT, kr=lambda s, a, b, r, c: _a4(s, a, b, r, 2 * c)
    ),
    "v_loop_variable_dropped": _mla_case(
        [(_MV, "n0[:, None] * vn")], MLA_FRAGMENT, v=lambda s, a, b, r, c: _a4(s, a, b, r % 32, c)
    ),
    "out_head_plus_1": _mla_case(
        [(_MO, _MO.replace("h * oh", "(h + 1) * oh"))], MLA_FRAGMENT, o=lambda s, a, b, r, c: _a4(s, a, b + 1, r, c)
    ),
    "batch_head_split_swapped": _mla_case(
        [("    z = hz // H\n    h = hz % H\n", "    z = hz % H\n    h = hz // H\n")],
        MLA_FRAGMENT,
        swap=True,
    ),
    # An equivalent spelling: must compute (the positional strides ran another program before).
    "eq_qn_batch_term_order": _mla_case([(_MQN, _MQN.replace("z * qnz + h * qnh", "h * qnh + z * qnz"))]),
    # The head index by the remainder's definition: must compute on the template.
    "eq_head_by_subtraction": _mla_case([("    h = hz % H\n", "    h = hz - z * H\n")]),
}


@pytest.fixture
def mla_hits(monkeypatch):
    import triton_msl.autotuning._fa_dispatch as fa_dispatch

    hits, real = [], fa_dispatch._dispatch_mla

    def spy(*args, **kwargs):
        result = real(*args, **kwargs)
        hits.append(bool(result))
        return result

    monkeypatch.setattr(fa_dispatch, "_dispatch_mla", spy)
    return hits


@requires_mps
@pytest.mark.parametrize("name", sorted(MLA_CASES))
def test_mla_variant_refuses_or_computes_the_source(name, mla_hits):
    from tests.test_mla_value_paths import _mla_value_path

    case = MLA_CASES[name]
    fn = _variant(_mla_value_path, *case["edits"])
    # Three batches over two heads: a swapped batch/head split reads head 2, outside the canonical view.
    z, heads, n_ctx, alloc = 3, 2, 64, 160
    widths = dict(qn=128, qr=64, kn=128, kr=64, v=128)
    gen = torch.Generator().manual_seed(1501)
    t = {n: (torch.randn(z + 1, heads + 1, alloc, w, generator=gen) * 0.5).half() for n, w in widths.items()}
    t["o"] = torch.full((z + 1, heads + 2, alloc, 128), POISON, dtype=torch.float16)

    def launch():
        dev = {n: x.to("mps") for n, x in t.items()}
        views = [dev[n][:z, :heads] for n in ("qn", "qr", "kn", "kr", "v", "o")]
        strides = [st for x in views for st in x.stride()]
        fn[(n_ctx // 32, z * heads)](*views, *strides, z, heads, n_ctx, 32, 32, 128, 64, 128, 0)
        torch.mps.synchronize()
        return [dev["o"]]

    def expected():
        s = {n: x.stride() for n, x in t.items()}
        exp = torch.full(t["o"].shape, POISON, dtype=torch.float64)
        rows = torch.arange(n_ctx)[:, None]
        for p in range(z * heads):
            a, b = (p % heads, p // heads) if case.get("swap") else (p // heads, p % heads)
            e = {}
            for n, w in widths.items():
                e[n] = _gather(t[n], case[n](s[n], a, b, rows, torch.arange(w)[None, :]))
            q = torch.cat([e["qn"], e["qr"]], -1) / 192**0.5
            k = torch.cat([e["kn"], e["kr"]], -1)
            values = torch.softmax(q @ k.T, -1) @ e["v"]
            _place(
                exp,
                case["o"](s["o"], a, b, rows, torch.arange(128)[None, :]),
                values,
                torch.ones(1, 1, dtype=torch.bool),
            )
        return [exp]

    verdict = _refused_or_computed(launch, expected, case["fragment"], mla_hits, tol=3e-2)
    if verdict == "computed" and case["fragment"] is None:
        assert mla_hits == [True]


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("name", ["col_and_row_nctx", "col_only_nctx", "lifted_col_nctx"])
def test_dense_column_clip_by_the_sequence_length_refuses_at_lowering(name, head_dim):
    # No device: the FA sequence length bounds ROWS only, so clipping head-dim columns with it
    # is a restricting mask the template would drop. Strides of 1 are specialized as the JIT does.
    from tests.test_flash_attention import _flash_attn_fwd
    from tests.test_template_scalar_abi import _lower

    fn = _variant(_flash_attn_fwd, *DENSE_CASES[name]["edits"])
    const = dict(BLOCK_M=32, BLOCK_N=32, HEAD_DIM=head_dim, IS_CAUSAL=False)
    const.update(stride_qk=1, stride_kk=1, stride_vk=1, stride_ok=1)
    sig = {n: "*fp32" if n in ("Q", "K", "V", "Out") else "i32" for n in fn.arg_names if n not in const}
    with pytest.raises(MetalNonRecoverableError, match=_FA_CLIP):
        _lower(fn, sig, const).lower()

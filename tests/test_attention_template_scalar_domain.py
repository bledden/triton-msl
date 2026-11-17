"""Attention templates read their arguments at the width and sign the source gives them, or refuse.

Every attention template (dense simdgroup and tiled, varlen, biased forward and backward, MLA)
binds its integer arguments as ``uint`` and adds its addresses in 32-bit arithmetic. The source
reads the same arguments as signed integers of their declared width. A negative stride from a
pointer at the last row of a view is a defined source launch (every address is inside the
allocation), and the templates ran it on other cells: inputs read from the wrong rows, outputs
stored outside the view. The lowering now names the arguments a template reads this way
(``attention_unsigned_scalars_v1``, the route's own launch contract riding inside) and the
launcher refuses a value the template would read differently. The varlen templates also read
the cumulative sequence lengths through ``int*`` whatever the tensor held: an int64 tensor
(what ``cumsum`` returns) gave every second entry the high word of the one before. The biased
template wrote the log-sum-exp through ``float*``: four bytes for every two of a half tensor.

Sections: the driver reason on synthetic launches (no device), the descriptors the lowering
emits (no device), and launches on the GPU: a negative stride refuses before anything runs, the
tiled dense template keeps its zero-copy route, int64 cumulative lengths compute, and the
log-sum-exp is stored at its tensor's element type.
"""

import math

import pytest
import torch
import triton

import tests.test_fa_bwd_address_witness as bwd
from tests.test_fa_biased_routing import _biased_fa
from tests.test_fa_launch_tile_contracts import FA_TAG, _assert_poisoned_match, _lower, _run_varlen, _variant
from tests.test_flash_attention import _flash_attn_fwd
from triton_msl.backend._launch_signature import scalar_bytes
from triton_msl.errors import MetalNonRecoverableError

requires_mps = pytest.mark.skipif(
    not (torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")),
    reason="needs MPS + compile_shader",
)

TAG = "attention_unsigned_scalars_v1"
POISON = -16384.0

# ---------------------------------------------------------------------------------------
# _attention_unsigned_scalars_reason on a synthetic launch: argument 0 is a pointer
# ---------------------------------------------------------------------------------------


_DEFAULT = object()


def _reason(values, declared="i32", unsigned=None, signed=(), payloads=None, descriptor=_DEFAULT, pointer=None):
    from triton_msl.backend.driver import _attention_unsigned_scalars_reason

    kargs = [object() if pointer is None else pointer, *values]
    signatures = ["*fp32", *([declared] * len(values) if isinstance(declared, str) else declared)]
    if payloads is None:
        payloads = [scalar_bytes(v, d) for v, d in zip(values, signatures[1:])]
    if unsigned is None:
        unsigned = tuple(i for i in range(1, len(kargs)) if i not in signed)
    if descriptor is _DEFAULT:
        descriptor = (TAG, tuple(unsigned), tuple(signed), None)
    return _attention_unsigned_scalars_reason(descriptor, kargs, signatures, [None, *payloads])


@pytest.mark.parametrize("values", [(64, 1, 4096), (0,), (2**31 - 1,)])
def test_values_the_template_reads_as_the_source_does_pass(values):
    assert _reason(values) is None


@pytest.mark.parametrize("value", [-1, -64, -(2**31)])
def test_negative_unsigned_read_refuses(value):
    reason = _reason((64, value))
    assert reason is not None
    assert f"argument 2 is {value}" in reason and "32-bit unsigned value" in reason
    assert "non-negative strides" in reason


@pytest.mark.parametrize("declared,value", [("i64", 2**31), ("i64", 2**32 + 64), ("u64", 2**63), ("i64", -(2**31) - 1)])
def test_value_outside_32_bits_refuses(declared, value):
    reason = _reason((value,), declared=declared)
    assert reason is not None and f"argument 1 is {value}" in reason


def test_u32_above_the_signed_range_refuses():
    reason = _reason((2**31,), declared="u32")
    assert reason is not None and "from 0 to 2**31 - 1" in reason


@pytest.mark.parametrize("value", [-5, 0, 64, -(2**31), 2**31 - 1])
def test_signed_read_takes_any_32_bit_value(value):
    assert _reason((64, value), signed=(2,)) is None


@pytest.mark.parametrize("declared,value", [("i64", 2**31), ("i64", -(2**31) - 1), ("u64", 2**63)])
def test_signed_read_outside_32_bits_refuses(declared, value):
    reason = _reason((64, value), declared=["i32", declared], signed=(2,))
    assert reason is not None and "32-bit signed value" in reason and "inside 32 bits" in reason


@pytest.mark.parametrize(
    "descriptor",
    [
        None,
        TAG,
        (TAG, (1,), ()),
        (TAG, (1,), (), None, None),
        (FA_TAG, (1,), (), None),
        (TAG, 1, (), None),
        (TAG, (1,), None, None),
        (TAG, (1.0,), (), None),
        (TAG, (True,), (), None),
        (TAG, (-1,), (), None),
        (TAG, (2,), (), None),
        (TAG, (), (7,), None),
    ],
)
def test_malformed_descriptor_refuses(descriptor):
    assert _reason((64,), descriptor=descriptor) == "the scalar-domain descriptor is missing or malformed"


def test_list_descriptor_is_accepted():
    assert _reason((64, -5), descriptor=[TAG, [1], [2], None]) is None
    assert _reason((-64, -5), descriptor=[TAG, [1], [2], None]) is not None


@pytest.mark.parametrize("declared", ["fp32", "*fp32", "i1", "i8", "i16"])
def test_non_integer_signature_refuses(declared):
    assert _reason((64,), declared=declared, payloads=[b"\0" * 4]) == "argument 1 is not an integer argument"


@pytest.mark.parametrize("payload", [None, b"", scalar_bytes(65, "i32"), bytearray(scalar_bytes(64, "i32"))])
def test_payload_mismatch_refuses(payload):
    assert _reason((64,), payloads=[payload]) == "argument 1 changed after its captured launch payload"


def test_reads_the_captured_value_not_a_subclass_int():
    class Lying(int):
        def __int__(self):
            return 64

    assert _reason((Lying(-64),)) is not None
    assert _reason((Lying(64),)) is None


def test_unevaluable_launch_refuses():
    from triton_msl.backend.driver import _attention_unsigned_scalars_reason

    reason = _attention_unsigned_scalars_reason((TAG, (1,), (), None), [object(), 64], ["*fp32"], [None, None])
    assert reason is not None and "could not be evaluated" in reason


def test_storage_beyond_the_32_bit_address_range_refuses(monkeypatch):
    import triton_msl.backend.driver as driver

    storage = torch.zeros(4096, dtype=torch.float16)
    assert _reason((64,), pointer=storage[100:200]) is None
    monkeypatch.setattr(driver, "_ATTENTION_TEMPLATE_ELEMENTS", 4096)
    assert _reason((64,), pointer=storage[100:200]) is None
    monkeypatch.setattr(driver, "_ATTENTION_TEMPLATE_ELEMENTS", 4095)
    reason = _reason((64,), pointer=storage[100:200])
    assert reason is not None and "argument 0 has a storage of 4096 elements" in reason


def test_address_range_limit_is_the_32_bit_index_range():
    from triton_msl.backend.driver import _ATTENTION_TEMPLATE_ELEMENTS

    assert _ATTENTION_TEMPLATE_ELEMENTS == 2**32


# ---------------------------------------------------------------------------------------
# Lowering: which arguments each template reads as unsigned 32-bit values
# ---------------------------------------------------------------------------------------


def _scalar_domain(lowerer):
    lowerer.lower()
    tag, unsigned, signed, _inner = lowerer._tail_access_bounds
    assert tag == TAG
    names = {a.index: a.name for a in lowerer.graph.args}
    return {names[i] for i in unsigned}, {names[i] for i in signed}


# Blocks of 32 x 32 take the simdgroup template, 64 x 64 the tiled one.
@pytest.mark.parametrize("head_dim,block", [(64, 32), (128, 32), (64, 64)])
def test_dense_lowering_names_every_stride_and_extent(head_dim, block):
    const = dict(BLOCK_M=block, BLOCK_N=block, HEAD_DIM=head_dim, IS_CAUSAL=False, Z=1)
    const.update(stride_qk=1, stride_kk=1, stride_vk=1, stride_ok=1)
    sig = {n: "*fp32" if n in ("Q", "K", "V", "Out") else "i32" for n in _flash_attn_fwd.arg_names if n not in const}
    unsigned, signed = _scalar_domain(_lower(_flash_attn_fwd, sig, const))
    strides = {f"stride_{t}{axis}" for t in "qkvo" for axis in ("z", "h")} | {
        "stride_qm", "stride_kn", "stride_vn", "stride_om",
    }  # fmt: skip
    assert unsigned == strides | {"H", "N_CTX"}
    assert signed == set()


def test_mla_lowering_names_every_stride_and_extent():
    from tests.test_mla_value_paths import _mla_value_path

    const = dict(BM=32, BN=32, DN=128, DR=64, DV=128, MODE=0, Z=1, qnk=1, qrk=1, knk=1, krk=1, vk=1, ok_=1)
    sig = {n: "*fp16" if n in ("Qn", "Qr", "Kn", "Kr", "V", "Out") else "i32" for n in _mla_value_path.arg_names}
    sig = {n: t for n, t in sig.items() if n not in const}
    unsigned, signed = _scalar_domain(_lower(_mla_value_path, sig, const))
    assert unsigned == {n for n, t in sig.items() if t == "i32"}
    assert signed == set()


# ---------------------------------------------------------------------------------------
# GPU: a negative row stride from a pointer at the last row refuses before anything runs
# ---------------------------------------------------------------------------------------


@pytest.fixture
def dispatches(monkeypatch):
    """Every attention replacement dispatch that ran (the driver imports them per call)."""
    import triton_msl.autotuning._fa_dispatch as fa_dispatch

    ran = []
    for name in ("dispatch_flash_attention", "_dispatch_mla"):

        def spy(*args, _real=getattr(fa_dispatch, name), _name=name, **kwargs):
            ran.append(_name)
            return _real(*args, **kwargs)

        monkeypatch.setattr(fa_dispatch, name, spy)
    return ran


def _from_last_row(t, axis, n):
    """(pointer, strides) addressing logical index r of ``axis`` at storage index n - 1 - r."""
    strides = list(t.stride())
    pointer = t.reshape(-1)[(n - 1) * strides[axis] :]
    strides[axis] = -strides[axis]
    return pointer, strides


def _dense(head_dim, dtype, block=32):
    n = 128
    t = {x: torch.randn(1, 2, n, head_dim).to(dtype) for x in "qkv"}
    t["o"] = torch.full((1, 2, n, head_dim), POISON, dtype=dtype)

    def launch(p, s):
        _flash_attn_fwd[(n // block, 2)](
            p["q"], p["k"], p["v"], p["o"], *s["q"], *s["k"], *s["v"], *s["o"], 1, 2, n,
            BLOCK_M=block, BLOCK_N=block, HEAD_DIM=head_dim, IS_CAUSAL=False,
        )  # fmt: skip

    return t, launch, {x: (2, n) for x in "qkvo"}


def _decode():
    """The inline small-query template: 17 query rows in 16-row tiles over 65 keys."""
    from tests.test_q1_decode_boundary import _q1_attention

    nq, nk, bm = 17, 65, 16
    rows = triton.cdiv(nq, bm) * bm
    t = {"q": torch.randn(1, 2, rows, 64), "k": torch.randn(1, 2, nk, 64), "v": torch.randn(1, 2, nk, 64)}
    t["o"] = torch.full((1, 2, rows, 64), POISON)

    def launch(p, s):
        _q1_attention[(rows // bm, 2)](
            p["q"], p["k"], p["v"], p["o"], *s["q"], *s["k"], *s["v"], *s["o"], 1, 2, nq, nk,
            BLOCK_M=bm, BLOCK_N=32, HEAD_DIM=64, IS_CAUSAL=False,
        )  # fmt: skip

    return t, launch, {"q": (2, rows), "k": (2, nk), "v": (2, nk), "o": (2, rows)}


def _varlen():
    from tests.test_varlen_fa_routing import _varlen_fwd

    heads, dim = 2, 64
    t = {"q": torch.randn(104, heads, dim), "k": torch.randn(114, heads, dim), "v": torch.randn(114, heads, dim)}
    t["o"] = torch.full((104, heads, dim), POISON)
    t["cu_q"] = torch.tensor([0, 40, 104], dtype=torch.int32)
    t["cu_k"] = torch.tensor([0, 50, 114], dtype=torch.int32)

    def launch(p, s):
        _varlen_fwd[(2, 2 * heads)](
            p["q"], p["k"], p["v"], p["o"], p["cu_q"], p["cu_k"],
            *s["q"], *s["k"], *s["v"], *s["o"], heads, 64, 1.0 / math.sqrt(dim), 32, 32, dim,
        )  # fmt: skip

    return t, launch, {"q": (0, 104), "k": (0, 114), "v": (0, 114), "o": (0, 104)}


def _biased():
    z, h, rows, dim = 1, 2, 64, 32
    t = {x: torch.randn(z, h, rows, dim) for x in "qkv"}
    t["b"] = torch.randn(z, h, rows, rows)
    t["m"] = (torch.rand(z, h, rows) < 0.25).to(torch.uint8)
    t["o"] = torch.full((z, h, rows, dim), POISON)
    t["l"] = torch.full((z, h, rows), POISON)

    def launch(p, s):
        args = []
        for x in ("o", "l", "q", "k", "v", "b", "m"):
            args += [p[x], *s[x]]
        _biased_fa[(rows // 32, z * h)](*args, dim**-0.5, -1e9, z, h, rows, dim, 32, 32)

    return t, launch, {"q": (2, rows), "k": (2, rows), "v": (2, rows), "b": (3, rows), "m": (2, rows), "o": (2, rows)}


def _mla():
    from tests.test_mla_value_paths import _mla_value_path

    z, nh, rows = 1, 2, 64
    names = ("qn", "qr", "kn", "kr", "v", "o")
    t = {n: torch.randn(z, nh, rows, d).half() for n, d in zip(names, (128, 64, 128, 64, 128))}
    t["o"] = torch.full((z, nh, rows, 128), POISON, dtype=torch.float16)

    def launch(p, s):
        _mla_value_path[(rows // 32, z * nh)](
            *[p[n] for n in names], *[x for n in names for x in s[n]], z, nh, rows, 32, 32, 128, 64, 128, 0
        )

    return t, launch, {n: (2, rows) for n in names}


def _backward(kind):
    t = bwd._inputs(kind, seed=1701)
    for name in bwd.OUTPUTS[kind]:
        t[name] = torch.full(t[name].shape, POISON)

    def launch(p, s):
        args = []
        for name in bwd.ORDER[kind]:
            args += [p[name], *s[name]]
        tiles = triton.cdiv(bwd.N, 32)
        if kind == "b":
            grid, tail = (tiles, tiles, bwd.HC), (bwd.SM, -1e9, bwd.HH, bwd.N, bwd.DIM, bwd.N, 32, 32)
        else:
            grid, tail = (tiles, bwd.INST, bwd.HC), (bwd.SM, -1e9, bwd.N, bwd.HH, bwd.DIM, bwd.N, 32, 32)
        bwd.KERNELS[kind][grid](*args, *tail)

    rows = {"kv": ("q", "k", "do", "dk", "dv"), "q": ("q", "k", "lse", "dq"), "b": ("q", "k", "do")}[kind]
    axes = {name: (2, bwd.N) for name in rows}
    if kind == "b":
        axes["db"] = (1, bwd.N)
    return t, launch, axes


ROUTES = {
    "dense-simd-64": (lambda: _dense(64, torch.float32), ("o",)),
    "dense-simd-128": (lambda: _dense(128, torch.float16), ("o",)),
    "dense-tiled-64": (lambda: _dense(64, torch.float32, 64), ("o",)),
    "decode": (_decode, ("o",)),
    "varlen": (_varlen, ("o",)),
    "biased": (_biased, ("o", "l")),
    "mla": (_mla, ("o",)),
    "backward-kv": (lambda: _backward("kv"), bwd.OUTPUTS["kv"]),
    "backward-q": (lambda: _backward("q"), bwd.OUTPUTS["q"]),
    "backward-b": (lambda: _backward("b"), bwd.OUTPUTS["b"]),
}
_NEGATIVE = [
    ("dense-simd-64", "q"), ("dense-simd-64", "k"), ("dense-simd-64", "v"), ("dense-simd-64", "o"),
    ("dense-simd-128", "k"), ("dense-simd-128", "o"),
    ("dense-tiled-64", "q"), ("dense-tiled-64", "k"), ("dense-tiled-64", "v"), ("dense-tiled-64", "o"),
    ("decode", "q"), ("decode", "k"), ("decode", "v"), ("decode", "o"),
    ("varlen", "q"), ("varlen", "k"), ("varlen", "v"), ("varlen", "o"),
    ("biased", "q"), ("biased", "k"), ("biased", "v"), ("biased", "b"), ("biased", "m"), ("biased", "o"),
    ("mla", "qn"), ("mla", "kr"), ("mla", "v"), ("mla", "o"),
    ("backward-kv", "q"), ("backward-kv", "k"), ("backward-kv", "do"), ("backward-kv", "dk"), ("backward-kv", "dv"),
    ("backward-q", "k"), ("backward-q", "lse"), ("backward-q", "dq"),
    ("backward-b", "q"), ("backward-b", "db"),
]  # fmt: skip


@requires_mps
@pytest.mark.parametrize("route,name", _NEGATIVE, ids=[f"{r}-{n}" for r, n in _NEGATIVE])
def test_negative_row_stride_refuses_at_launch(route, name, dispatches):
    make, outputs = ROUTES[route]
    torch.manual_seed(1723)
    tensors, launch, axes = make()
    device = {key: x.to("mps") for key, x in tensors.items()}
    pointers, strides = dict(device), {key: list(x.stride()) for key, x in device.items()}
    pointers[name], strides[name] = _from_last_row(device[name], *axes[name])
    with pytest.raises(MetalNonRecoverableError, match="Refusing attention template launch") as info:
        launch(pointers, strides)
    assert "32-bit unsigned value" in str(info.value) and "non-negative strides" in str(info.value)
    assert dispatches == []
    for out in outputs:
        assert bool((device[out].cpu() == POISON).all())


@requires_mps
@pytest.mark.parametrize("route", sorted(ROUTES))
def test_ordinary_strides_still_launch(route, dispatches):
    make, outputs = ROUTES[route]
    torch.manual_seed(1724)
    tensors, launch, _axes = make()
    device = {key: x.to("mps") for key, x in tensors.items()}
    launch(dict(device), {key: list(x.stride()) for key, x in device.items()})
    torch.mps.synchronize()
    assert bool((device[outputs[0]].cpu() != POISON).any())
    # The tiled dense and small-query templates have no replacement dispatch: they are
    # inline shaders on the generic 2-D route.
    assert bool(dispatches) == (route not in ("dense-tiled-64", "decode"))


@requires_mps
def test_tiled_dense_template_keeps_the_zero_copy_route(monkeypatch):
    import triton_msl.backend._generic_launch as generic_launch

    geometries, real = [], generic_launch.generic_2d_geometry

    def spy(*args, **kwargs):
        geometries.append(real(*args, **kwargs))
        return geometries[-1]

    monkeypatch.setattr(generic_launch, "generic_2d_geometry", spy)
    torch.manual_seed(1725)
    tensors, launch, _axes = _dense(64, torch.float32, 64)
    device = {key: x.to("mps") for key, x in tensors.items()}
    launch(dict(device), {key: list(x.stride()) for key, x in device.items()})
    torch.mps.synchronize()
    assert len(geometries) == 1 and geometries[0] is not None
    torch.testing.assert_close(device["o"].cpu().double(), _dense_reference(tensors), atol=2e-3, rtol=2e-3)


def _dense_reference(tensors, flip=None):
    q, k, v = (tensors[x].double().flip(2) if x == flip else tensors[x].double() for x in "qkv")
    return torch.softmax((q * q.shape[-1] ** -0.5) @ k.transpose(-1, -2), -1) @ v


@requires_mps
@pytest.mark.parametrize("name", ["q", "k", "v", "o"])
def test_generic_dense_attention_computes_a_negative_row_stride(name, dispatches):
    # head_dim 32 takes no template: the generic lowering reads the strides as the source does.
    torch.manual_seed(1727)
    tensors, launch, axes = _dense(32, torch.float32)
    device = {key: x.to("mps") for key, x in tensors.items()}
    pointers, strides = dict(device), {key: list(x.stride()) for key, x in device.items()}
    pointers[name], strides[name] = _from_last_row(device[name], *axes[name])
    launch(pointers, strides)
    torch.mps.synchronize()
    got = device["o"].cpu().double()
    torch.testing.assert_close(
        got.flip(2) if name == "o" else got, _dense_reference(tensors, name), atol=2e-3, rtol=2e-3
    )
    assert dispatches == []


@requires_mps
def test_unmasked_dense_store_with_a_negative_stride_refuses_before_the_tile_contract(dispatches):
    # Both contracts apply to this kernel; the scalar domain is checked first.
    fn = _variant(
        _flash_attn_fwd,
        (
            "tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N_CTX)",
            "tl.store(o_ptrs, acc.to(Out.dtype.element_ty))",
        ),
    )
    n, d = 64, 64
    torch.manual_seed(1726)
    q, k, v = [torch.randn(1, 2, n, d, device="mps") for _ in range(3)]
    out = torch.full((1, 2, n, d), POISON, device="mps")
    pointer, strides = _from_last_row(q, 2, n)
    with pytest.raises(MetalNonRecoverableError, match="Refusing attention template launch"):
        fn[(n // 32, 2)](
            pointer, k, v, out, *strides, *k.stride(), *v.stride(), *out.stride(), 1, 2, n,
            BLOCK_M=32, BLOCK_N=32, HEAD_DIM=d, IS_CAUSAL=False,
        )  # fmt: skip
    assert dispatches == []
    assert bool((out.cpu() == POISON).all())


# ---------------------------------------------------------------------------------------
# GPU: varlen cumulative sequence lengths are read at their stored width
# ---------------------------------------------------------------------------------------


@requires_mps
@pytest.mark.parametrize(
    "cu_dtypes", [(torch.int64, torch.int64), (torch.int32, torch.int64), (torch.int64, torch.int32)], ids=str
)
# fp32 at head_dim 128 exceeds the simdgroup tile budget and takes the scalar varlen template.
@pytest.mark.parametrize("dim,dtype", [(64, torch.float32), (128, torch.float16), (128, torch.float32)])
def test_varlen_int64_cumulative_lengths_compute(dim, dtype, cu_dtypes, dispatches):
    out, expected = _run_varlen([40, 64, 7], [50, 64, 33], 64, dim, dtype, cu_dtypes)
    assert dispatches == ["dispatch_flash_attention"]
    _assert_poisoned_match(out, expected, 2e-2 if dtype == torch.float16 else 1e-3)


# ---------------------------------------------------------------------------------------
# The biased log-sum-exp is stored at its tensor's element type
# ---------------------------------------------------------------------------------------

_FLOATS = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
_MSL = {"fp32": "float", "fp16": "half", "bf16": "bfloat"}


@pytest.mark.parametrize("lse", sorted(_FLOATS))
def test_biased_lowering_declares_the_lse_element_type(lse):
    from triton_msl.codegen._msl_templates import _lse_store

    assert _lse_store("x", _MSL[lse]) == ("x" if lse == "fp32" else f"{_MSL[lse]}(x)")
    with pytest.raises(ValueError, match="float/half/bfloat"):
        _lse_store("x", "double")


@requires_mps
@pytest.mark.parametrize("lse", sorted(_FLOATS))
@pytest.mark.parametrize("base", sorted(_FLOATS))  # fp32 and fp16: simdgroup template; bf16: tiled
def test_biased_lse_is_stored_at_its_element_type(base, lse, dispatches):
    z, h, rows, dim = 1, 2, 64, 32
    gen = torch.Generator().manual_seed(1723)
    q, k, v = (torch.randn(z, h, rows, dim, generator=gen).to(_FLOATS[base]) for _ in range(3))
    bias = torch.randn(z, h, rows, rows, generator=gen).to(_FLOATS[base])
    mask = torch.rand(z, h, rows, generator=gen) < 0.25
    out = torch.full((z, h, rows, dim), POISON, dtype=_FLOATS[base], device="mps")
    # Twice the rows the kernel stores: a store wider than the element runs into the second half.
    lse_storage = torch.full((z, h, 2 * rows), POISON, dtype=_FLOATS[lse], device="mps")
    args = []
    for x in (out, lse_storage, q.to("mps"), k.to("mps"), v.to("mps"), bias.to("mps"), mask.to(torch.uint8).to("mps")):
        args += [x, *x.stride()]
    _biased_fa[(rows // 32, z * h)](*args, dim**-0.5, -1e9, z, h, rows, dim, 32, 32)
    torch.mps.synchronize()
    assert dispatches == ["dispatch_flash_attention"]
    raw = dim**-0.5 * (q.double() @ k.double().transpose(-1, -2)) + bias.double()
    raw = raw.masked_fill(mask[:, :, None, :], float("-inf"))
    expected = torch.full((z, h, 2 * rows), POISON, dtype=torch.float64)
    expected[:, :, :rows] = torch.logsumexp(raw, -1)
    tol = 4e-2 if "bf16" in (base, lse) else 1e-2 if "fp16" in (base, lse) else 1e-4
    _assert_poisoned_match(lse_storage, expected, tol)
    _assert_poisoned_match(out, torch.softmax(raw, -1) @ v.double(), tol)

"""Every attention route with some of its tensors laid out as blocks of one storage.

An attention launch whose output is a view of the storage of another tensor argument runs
only when the launcher shows the two apart. The views a caller passes do not show it
(tests/test_attention_output_alias.py states why), so the launcher bounds the cells the
source reaches through each tensor argument from the launch's own strides, lengths and grid
(tests/test_address_extent.py), and admits the launch when every bound lies inside its
storage and no cell stored through one argument is reached through another. Tensors as
contiguous blocks of one storage, each passed with its own shape and strides, are such a
layout.

For each route (dense forward, decode, varlen, MLA, biased with its output and lse, and the
three backward kernels) and each layout:

* every tensor in a storage of its own computes a float64 reference, on the template the
  route publishes;
* an output and another tensor as blocks of one storage, a few cells apart, compute that
  reference on the same template, and no cell outside the outputs changes; so do two
  outputs, and so do host tensors on the routes that take them;
* blocks laid so that each starts on the cell after the last one reached through the block
  before it compute, and laid one cell closer they refuse by the two argument names, before
  any launch, and no cell of the storage changes;
* what a forward output held before the launch does not reach the result;
* inputs alone in one storage, the outputs apart, compute what separate storages compute.

The varlen route takes each sequence's first row from a tensor, which the launcher does not
read, so the cells of a varlen launch are not bounded and its output beside another tensor
in one storage still refuses. Those layouts computed before the storage rule: their refusal
is a capability regression, and not a wrong result that was fixed.

The last test pins a dense kernel whose output base is chosen by a scalar branch. It
publishes no outputs and is not replaced by a template, so the storage refusal cannot fire
for it. With the output over the keys the launcher refuses it for the lane order instead.
"""

import dataclasses
import itertools
import math
import platform
import types
from collections.abc import Callable

import pytest
import torch
import triton

import tests.test_fa_bwd_address_witness as bwd
import triton_msl.backend._address_extent as address_extent
from tests.test_fa_biased_routing import _biased_fa
from tests.test_flash_attention import _flash_attn_fwd
from tests.test_mla_value_paths import _mla_expected, _mla_value_path
from tests.test_q1_decode_boundary import _q1_attention
from tests.test_varlen_fa_routing import _ref_varlen, _varlen_fwd
from triton_msl.errors import MetalNonRecoverableError

pytestmark = pytest.mark.skipif(
    not (platform.system() == "Darwin" and torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")),
    reason="needs MPS + compile_shader",
)

POISON = -16384.0  # fills the forward outputs and the cells between the blocks of a storage
GAP = 7  # cells between two blocks of one storage
GUARD = 64  # cells after its last block
REFUSAL = "attention kernel: output argument {} is not known to be apart from argument {}:"
MMA = "simdgroup-MMA FlashAttention-2"
TILED = "Head-dim-tiled FlashAttention-2"


@dataclasses.dataclass(frozen=True)
class Case:
    """One attention kernel at one size."""

    label: str
    pointers: dict  # tensor name -> the kernel's argument name, in argument order
    outputs: tuple  # the tensors the kernel stores through, in argument order
    stored: tuple  # their positions among the launch arguments, as the kernel publishes them
    descriptor: str | None  # tag of the template the kernel publishes; None when its own shader computes
    template: str | None  # a line of the template that computes the launch; None for the kernel as lowered
    tolerance: float  # absolute; per unit of the largest expected value when ``relative`` is set
    make: Callable  # seed -> host tensors, the outputs filled with ``unwritten``
    launch: Callable  # device tensors -> the compiled kernel
    reference: Callable  # host tensors -> float64 result per output, ``unwritten`` where nothing is stored
    shared: tuple  # groups of tensors with an output among them
    inputs: tuple  # a group of inputs only
    unwritten: float = POISON
    relative: float = 0.0
    bounded: bool = True  # the launch's scalars bound the cells it reaches
    on_host: bool = True  # the route takes host tensors


def _strides(t, names):
    return [stride for name in names for stride in t[name].stride()]


def _dense(dim, dtype, n, rows, heads, descriptor, template):
    def make(seed):
        gen = torch.Generator().manual_seed(seed)
        t = {name: torch.randn(1, heads, n, dim, generator=gen).to(dtype) for name in "qkv"}
        return dict(t, o=torch.full((1, heads, n, dim), POISON, dtype=dtype))

    def launch(t):
        return _flash_attn_fwd[(triton.cdiv(n, rows), heads)](
            *(t[name] for name in "qkvo"),
            *_strides(t, "qkvo"),
            1,
            heads,
            n,
            BLOCK_M=rows,
            BLOCK_N=32,
            HEAD_DIM=dim,
            IS_CAUSAL=False,
        )

    def reference(t):
        # The source loads whole key blocks with zeros past the last key, and scores them.
        pad = (0, 0, 0, triton.cdiv(n, 32) * 32 - n)
        k, v = (torch.nn.functional.pad(t[name].double(), pad) for name in "kv")
        return {"o": torch.softmax(t["q"].double() @ k.transpose(-1, -2) / math.sqrt(dim), -1) @ v}

    half = dtype == torch.float16
    return Case(
        label=f"dense-d{dim}-{'fp16' if half else 'fp32'}-n{n}-rows{rows}",
        pointers={"q": "Q", "k": "K", "v": "V", "o": "Out"},
        outputs=("o",),
        stored=(3,),
        descriptor=descriptor,
        template=template,
        tolerance=2e-2 if half else 1e-3,
        make=make,
        launch=launch,
        reference=reference,
        shared=(("q", "o"), ("o", "k"), ("v", "o"), ("q", "k", "v", "o")),
        inputs=("q", "k", "v"),
    )


def _decode(dtype, rows, queries, keys):
    heads = 2

    def make(seed):
        gen = torch.Generator().manual_seed(seed)
        t = {"q": torch.randn(1, heads, queries, 64, generator=gen).to(dtype)}
        t["k"] = torch.randn(1, heads, keys, 64, generator=gen).to(dtype)
        t["v"] = torch.randn(1, heads, keys, 64, generator=gen).to(dtype)
        return dict(t, o=torch.full((1, heads, queries, 64), POISON, dtype=dtype))

    def launch(t):
        return _q1_attention[(triton.cdiv(queries, rows), heads)](
            *(t[name] for name in "qkvo"),
            *_strides(t, "qkvo"),
            1,
            heads,
            queries,
            keys,
            BLOCK_M=rows,
            BLOCK_N=32,
            HEAD_DIM=64,
            IS_CAUSAL=False,
        )

    def reference(t):
        # The source rounds the scaled query back to its dtype, and loads whole key blocks
        # with zeros past the last key (tests/test_q1_decode_boundary.py).
        scaled = (t["q"].float() * 0.125).to(dtype).double()
        pad = (0, 0, 0, triton.cdiv(keys, 32) * 32 - keys)
        k, v = (torch.nn.functional.pad(t[name].double(), pad) for name in "kv")
        return {"o": torch.softmax(scaled @ k.transpose(-1, -2), -1) @ v}

    half = dtype == torch.float16
    return Case(
        label=f"decode-{'fp16' if half else 'fp32'}-q{queries}-k{keys}-rows{rows}",
        pointers={"q": "Q", "k": "K", "v": "V", "o": "Out"},
        outputs=("o",),
        stored=(3,),
        descriptor=None,
        template=TILED,
        tolerance=4e-3 if half else 1e-5,
        make=make,
        launch=launch,
        reference=reference,
        shared=(("q", "o"), ("o", "k"), ("v", "o")),
        inputs=("k", "v"),
    )


def _varlen(dim, lengths):
    heads, total, longest = 2, sum(lengths), max(lengths)
    starts = torch.tensor([0, *itertools.accumulate(lengths)], dtype=torch.int32)

    def make(seed):
        gen = torch.Generator().manual_seed(seed)
        t = {name: torch.randn(total, heads, dim, generator=gen) for name in "qkv"}
        return dict(t, cq=starts.clone(), ck=starts.clone(), o=torch.full((total, heads, dim), POISON))

    def launch(t):
        return _varlen_fwd[(triton.cdiv(longest, 32), len(lengths) * heads)](
            *(t[name] for name in ("q", "k", "v", "o", "cq", "ck")),
            *_strides(t, "qkvo"),
            heads,
            longest,
            1.0 / math.sqrt(dim),
            32,
            32,
            dim,
        )

    def reference(t):
        return {"o": _ref_varlen(t["q"], t["k"], t["v"], t["cq"], t["ck"], heads, dim, 1.0 / math.sqrt(dim)).double()}

    return Case(
        label=f"varlen-d{dim}-" + "-".join(str(length) for length in lengths),
        pointers={"q": "Q", "k": "K", "v": "V", "o": "Out", "cq": "cu_q", "ck": "cu_k"},
        outputs=("o",),
        stored=(3,),
        descriptor="flash_attention",
        template="Varlen FlashAttention-2",
        tolerance=1e-3,
        make=make,
        launch=launch,
        reference=reference,
        shared=(("q", "o"), ("o", "k"), ("v", "o")),
        inputs=("cq", "ck"),
        bounded=False,  # each sequence's first row is read from cu_q and cu_k
    )


def _mla(rows):
    heads, names = 2, ("qn", "qr", "kn", "kr", "v", "o")
    widths = {"qn": 128, "qr": 64, "kn": 128, "kr": 64, "v": 128}

    def make(seed):
        gen = torch.Generator().manual_seed(seed)
        t = {name: (torch.randn(1, heads, rows, width, generator=gen) * 0.5).half() for name, width in widths.items()}
        return dict(t, o=torch.full((1, heads, rows, 128), POISON, dtype=torch.float16))

    def launch(t):
        return _mla_value_path[(triton.cdiv(rows, 32), heads)](
            *(t[name] for name in names), *_strides(t, names), 1, heads, rows, 32, 32, 128, 64, 128, 0
        )

    def reference(t):
        return {"o": _mla_expected(t["qn"], t["qr"], t["kn"], t["kr"], t["v"], 0).double()}

    return Case(
        label=f"mla-rows{rows}",
        pointers={"qn": "Qn", "qr": "Qr", "kn": "Kn", "kr": "Kr", "v": "V", "o": "Out"},
        outputs=("o",),
        stored=(5,),
        descriptor="mla",
        template=MMA,
        tolerance=5e-2,
        make=make,
        launch=launch,
        reference=reference,
        shared=(("qn", "o"), ("o", "kr"), ("v", "o")),
        inputs=("qn", "kn", "v"),
        on_host=False,
    )


def _biased(dim, dtype, rows, keys_block, template):
    heads, names = 2, ("o", "l", "q", "k", "v", "b", "m")

    def make(seed):
        gen = torch.Generator().manual_seed(seed)
        t = {name: torch.randn(1, heads, rows, dim, generator=gen).to(dtype) for name in "qkv"}
        t["b"] = torch.randn(1, heads, rows, rows, generator=gen).to(dtype)
        t["m"] = (torch.rand(1, heads, rows, generator=gen) < 0.25).to(torch.uint8)
        t["o"] = torch.full((1, heads, rows, dim), POISON, dtype=dtype)
        return dict(t, l=torch.full((1, heads, rows), POISON, dtype=dtype))

    def launch(t):
        args = []
        for name in names:
            args += [t[name], *t[name].stride()]
        return _biased_fa[(triton.cdiv(rows, 32), heads)](*args, dim**-0.5, -1e9, 1, heads, rows, dim, 32, keys_block)

    def reference(t):
        raw = dim**-0.5 * (t["q"].double() @ t["k"].double().transpose(-1, -2)) + t["b"].double()
        raw = raw.masked_fill(t["m"][:, :, None, :].bool(), float("-inf"))
        return {"o": torch.softmax(raw, -1) @ t["v"].double(), "l": torch.logsumexp(raw, -1)}

    half = dtype == torch.float16
    pointers = {"o": "o_ptr", "l": "lse_ptr", "q": "q_ptr", "k": "k_ptr", "v": "v_ptr", "b": "b_ptr", "m": "mask_ptr"}
    return Case(
        label=f"biased-d{dim}-{'fp16' if half else 'fp32'}-rows{rows}-keys{keys_block}",
        pointers=pointers,
        outputs=("o", "l"),
        stored=(0, 4),
        descriptor="flash_attention",
        template=template,
        tolerance=3e-2 if half else 1e-3,
        make=make,
        launch=launch,
        reference=reference,
        shared=(("q", "o"), ("b", "l"), ("o", "v"), ("o", "l"), ("l", "o")),
        inputs=("q", "k", "v", "b"),
    )


def _backward(kind, stored, template, shared):
    names = bwd.ORDER[kind]
    if kind == "b":
        grid = (triton.cdiv(bwd.N, 32), triton.cdiv(bwd.N, 32), bwd.HC)
        tail = (bwd.SM, -1e9, bwd.HH, bwd.N, bwd.DIM, bwd.N, 32, 32)
    else:
        grid = (triton.cdiv(bwd.N, 32), bwd.INST, bwd.HC)
        tail = (bwd.SM, -1e9, bwd.N, bwd.HH, bwd.DIM, bwd.N, 32, 32)

    def launch(t):
        args = []
        for name in names:
            args += [t[name], *t[name].stride()]
        return bwd.KERNELS[kind][grid](*args, *tail)

    def reference(t):
        # The witness's float64 evaluation of the source at its own addresses; every tensor
        # is larger than its logical shape, and the cells no program stores keep the fill.
        return dict(zip(bwd.OUTPUTS[kind], bwd._expected(kind, bwd._case(), t)))

    renamed = {"delta": "d_ptr", "lse": "l_ptr"}  # the other pointers are named <tensor>_ptr
    return Case(
        label=f"backward-{kind}",
        pointers={name: renamed.get(name, name + "_ptr") for name in names},
        outputs=tuple(name for name in names if name in bwd.OUTPUTS[kind]),
        stored=stored,
        descriptor="flash_attention",
        template=template,
        tolerance=2e-4,
        make=lambda seed: bwd._inputs(kind, seed),
        launch=launch,
        reference=reference,
        shared=shared,
        inputs=("q", "k", "v"),
        unwritten=bwd.POISON,
        relative=2e-3,
    )


CASES = [
    _dense(64, torch.float32, 80, 32, 2, "flash_attention", MMA),
    _dense(64, torch.float16, 64, 64, 1, None, TILED),
    _dense(32, torch.float32, 96, 32, 3, None, None),  # head dim 32 has no template: the kernel as lowered
    _dense(128, torch.float16, 64, 32, 2, "flash_attention", MMA),
    _decode(torch.float16, 8, 1, 96),
    _decode(torch.float32, 16, 16, 64),
    _decode(torch.float32, 8, 1, 70),
    _varlen(64, [20, 32]),
    _varlen(32, [7, 45, 33]),
    _mla(32),
    _mla(64),
    _biased(32, torch.float32, 32, 32, MMA),
    _biased(64, torch.float16, 32, 32, TILED),
    _biased(64, torch.float32, 64, 64, MMA),
    _backward(
        "q",
        (0, 32),
        "FA-2 backward dQ + delta",
        (("q", "dq"), ("lse", "delta"), ("do", "dq"), ("dq", "delta")),
    ),
    _backward("kv", (28, 32), "FA-2 backward dK/dV", (("k", "dk"), ("dv", "v"), ("delta", "dk"), ("dk", "dv"))),
    _backward("b", (28,), "FA-2 backward dbias", (("b", "db"), ("db", "delta"))),
]
GROUPS = [(case, group) for case in CASES for group in case.shared]
BOUNDED = [(case, group) for case, group in GROUPS if case.bounded]
UNBOUNDED = [(case, group) for case, group in GROUPS if not case.bounded]
FORWARD = [(case, group) for case, group in BOUNDED if not case.label.startswith("backward")]
ON_HOST = [(case, group) for case, group in BOUNDED if case.on_host]


def _label(value):
    return value.label if isinstance(value, Case) else "+".join(value)


@pytest.fixture
def launched(monkeypatch):
    """What ran: ``templates`` holds the results of the attention template dispatch,
    ``shaders`` the shader entry points launched."""
    import triton_msl.autotuning._fa_dispatch as fa_dispatch
    from tests.cache_helpers import patch_live_singleton_method
    from triton_msl.backend.driver import _get_compile_shader_runtime

    seen = types.SimpleNamespace(templates=[], shaders=[])
    real_template = fa_dispatch.dispatch_flash_attention

    def spy_template(*args, **kwargs):
        result = real_template(*args, **kwargs)
        seen.templates.append(bool(result))
        return result

    def observed(_runtime, launch):
        def spy_dispatch(lib, kernel_name, args, **kwargs):
            seen.shaders.append(kernel_name)
            return launch(lib, kernel_name, args, **kwargs)

        return spy_dispatch

    monkeypatch.setattr(fa_dispatch, "dispatch_flash_attention", spy_template)
    # The runtime is one object for the process, and a test that ran earlier may have left
    # its own ``dispatch`` on it, which a spy on the class would never see.
    patch_live_singleton_method(monkeypatch, _get_compile_shader_runtime, "dispatch", observed)
    return seen


@pytest.fixture
def bound(monkeypatch):
    """What the launcher bounded last: ``extents`` holds the least and the most cell reached
    through each tensor argument, by its position among ``arguments``."""
    seen = types.SimpleNamespace(extents=None, arguments=None)
    real = address_extent.reached

    def spy(plan, arguments, grid):
        out = real(plan, arguments, grid)
        seen.extents = {index: (low, high) for index, (low, high, _stored) in out.items()}
        seen.arguments = list(arguments)
        return out

    monkeypatch.setattr(address_extent, "reached", spy)
    return seen


def _on_device(host, device="mps"):
    return {name: x.clone().to(device) for name, x in host.items()}


def _place(host, group, starts, device="mps"):
    """The tensors of a launch with those of ``group`` as contiguous blocks of one storage,
    each from its own first cell and with its own shape and strides. Every block lies whole
    inside the storage."""
    dtype = host[group[0]].dtype
    assert all(host[name].dtype == dtype for name in group)
    fill = POISON if dtype.is_floating_point else 0
    cells = max(starts[name] + host[name].numel() for name in group) + GUARD
    storage = torch.full((cells,), fill, dtype=dtype, device=device)
    t = _on_device(host, device)
    for name in group:
        t[name] = storage[starts[name] : starts[name] + host[name].numel()].view(host[name].shape)
        t[name].copy_(host[name])
    return t, storage


def _in_one_storage(host, group, gap=GAP, device="mps"):
    """The tensors of a launch with those of ``group`` laid in the group's order, ``gap``
    cells apart, and the cells of each block in the storage."""
    starts, at = {}, 0
    for name in group:
        starts[name] = at
        at += host[name].numel() + gap
    t, storage = _place(host, group, starts, device)
    return t, storage, {name: (starts[name], starts[name] + host[name].numel()) for name in group}


def _run(case, t):
    handle = case.launch(t)
    torch.mps.synchronize()
    return handle


def _assert_route(case, handle, launched):
    """The launch ran the shader the route publishes, once, and nothing else."""
    published = handle.metadata.flash_attention
    assert tuple(handle.metadata.attention_outputs) == case.stored
    if case.descriptor is None:
        assert published is None and launched.templates == []
        source = handle.asm["msl"]
    else:
        assert published[0] == case.descriptor and launched.templates == [True]
        source = published[1]
    if case.template is None:
        assert MMA not in source and TILED not in source
    else:
        assert case.template in source
    assert len(launched.shaders) == 1


def _assert_close(case, got, expected, name):
    written = expected != case.unwritten
    assert torch.equal(got != case.unwritten, written), name
    scale = max(1.0, float(expected[written].abs().max())) if case.relative and bool(written.any()) else 1.0
    torch.testing.assert_close(got[written], expected[written], atol=case.tolerance * scale, rtol=case.relative)


def _assert_reference(case, host, t):
    for name, expected in case.reference(host).items():
        _assert_close(case, t[name].cpu().double(), expected, name)


def _assert_the_rest_kept(case, group, storage, before, spans):
    """No cell of the storage outside the blocks of its outputs changed."""
    kept = torch.ones(storage.numel(), dtype=torch.bool)
    for name in group:
        if name in case.outputs:
            kept[slice(*spans[name])] = False
    assert torch.equal(storage.cpu()[kept], before.cpu()[kept])


def _refusal(case, group):
    """The two argument names of the refusal: the first output of the group in argument
    order, then the first other tensor of the group in argument order."""
    output = next(name for name in case.pointers if name in group and name in case.outputs)
    other = next(name for name in case.pointers if name in group and name != output)
    return REFUSAL.format(case.pointers[output], case.pointers[other])


def _assert_refused(case, group, launched, t, storage):
    before = storage.clone()
    with pytest.raises(MetalNonRecoverableError, match=_refusal(case, group)):
        case.launch(t)
    torch.mps.synchronize()
    assert torch.equal(storage, before)
    assert launched.templates == [] and launched.shaders == []


@pytest.mark.parametrize("case", CASES, ids=_label)
def test_separate_storages_compute_the_reference(case, launched):
    host = case.make(3)
    t = _on_device(host)
    handle = _run(case, t)
    _assert_route(case, handle, launched)
    _assert_reference(case, host, t)


@pytest.mark.parametrize("case, group", BOUNDED, ids=_label)
def test_blocks_of_one_storage_with_an_output_among_them_compute_the_reference(case, group, launched):
    host = case.make(3)
    t, storage, spans = _in_one_storage(host, group)
    before = storage.clone()
    handle = _run(case, t)
    _assert_route(case, handle, launched)
    _assert_reference(case, host, t)
    _assert_the_rest_kept(case, group, storage, before, spans)


@pytest.mark.parametrize("case, group", ON_HOST, ids=_label)
def test_blocks_of_one_host_storage_compute_the_reference(case, group):
    host = case.make(3)
    t, storage, spans = _in_one_storage(host, group, device="cpu")
    before = storage.clone()
    _run(case, t)
    _assert_reference(case, host, t)
    _assert_the_rest_kept(case, group, storage, before, spans)


@pytest.mark.parametrize("case, group", BOUNDED, ids=_label)
def test_blocks_that_touch_by_the_cells_reached_compute_and_one_cell_closer_refuse(case, group, launched, bound):
    """The launcher compares the cells a launch reaches, not the views it is passed. The
    backward tensors are larger than the cells their kernels reach, so their blocks overlap
    as views here and compute."""
    host = case.make(3)
    t, _, _ = _in_one_storage(host, group)
    _run(case, t)
    last = {}
    for name in group:
        index = next(i for i, value in enumerate(bound.arguments) if value is t[name])
        least, last[name] = bound.extents[index]
        assert least == 0 and last[name] < host[name].numel(), name
    for closer in (0, 1):
        launched.templates.clear()
        launched.shaders.clear()
        starts, at = {}, 0
        for name in group:
            starts[name] = at
            at += last[name] + 1 - closer
        t, storage = _place(host, group, starts)
        if closer:
            _assert_refused(case, group, launched, t, storage)
            continue
        handle = _run(case, t)
        _assert_route(case, handle, launched)
        expected = case.reference(host)
        for name in case.pointers:
            # The cells of a block past the last one reached belong to the block after it.
            cells = last[name] + 1 if name in group[:-1] else host[name].numel()
            got = t[name].cpu().double().reshape(-1)[:cells]
            if name in expected:
                _assert_close(case, got, expected[name].reshape(-1)[:cells], name)
            else:
                assert torch.equal(got, host[name].double().reshape(-1)[:cells]), name


@pytest.mark.parametrize("case, group", FORWARD, ids=_label)
def test_what_an_output_held_before_the_launch_does_not_reach_the_result(case, group, launched):
    """The blocks touch and every output holds NaN before the launch, so a result computed
    from a cell of an output would hold one."""
    host = case.make(3)
    t, _, _ = _in_one_storage(host, group, gap=0)
    for name in case.outputs:
        t[name].fill_(float("nan"))
    handle = _run(case, t)
    _assert_route(case, handle, launched)
    _assert_reference(case, host, t)


@pytest.mark.parametrize("case, group", UNBOUNDED, ids=_label)
def test_an_output_beside_another_tensor_refuses_when_a_tensor_places_the_rows(case, group, launched):
    """A capability regression, not a wrong that was fixed: the output and the other tensor
    are blocks of one storage that share no cell, and this layout computed correctly before
    the storage rule. The launcher does not read the tensor that holds each sequence's first
    row, so it cannot see that the kernel stays inside the views it is passed."""
    t, storage, _ = _in_one_storage(case.make(3), group)
    _assert_refused(case, group, launched, t, storage)


@pytest.mark.parametrize("case", CASES, ids=_label)
def test_inputs_in_one_storage_compute_the_separate_result(case, launched):
    host = case.make(3)
    apart = _on_device(host)
    _run(case, apart)
    launched.templates.clear()
    launched.shaders.clear()
    t, storage, _ = _in_one_storage(host, case.inputs)
    before = storage.clone()
    handle = _run(case, t)
    _assert_route(case, handle, launched)
    assert torch.equal(storage, before)
    for name in case.outputs:
        assert torch.equal(t[name], apart[name]), name
    _assert_reference(case, host, t)


def test_an_output_base_chosen_by_a_scalar_branch_is_refused_in_place_by_the_lane_order(launched):
    """The dense kernel with its output base chosen by ``if N_CTX > 16`` publishes no
    outputs and is not replaced by a template, so the storage refusal cannot fire for it.
    The kernel as lowered computes with a separate output, and with the output over the
    keys the launcher refuses it for the lane order instead, before any launch."""
    chosen = bwd._variant(
        _flash_attn_fwd,
        (
            "    o_ptrs = Out + off_z * stride_oz",
            "    if N_CTX > 16:\n        base = Out\n    else:\n        base = Out + N_CTX\n"
            "    o_ptrs = base + off_z * stride_oz",
        ),
    )
    n, dim = 32, 64
    strides = (n * dim, n * dim, dim, 1)
    gen = torch.Generator().manual_seed(100)
    q, k, v = (torch.randn(1, 1, n, dim, generator=gen) for _ in range(3))
    expected = torch.softmax(q.double() @ k.double().transpose(-1, -2) / math.sqrt(dim), -1) @ v.double()
    storage = torch.full((4 * n * dim,), POISON, device="mps")
    keys = storage[: n * dim].view(1, 1, n, dim)
    keys.copy_(k)

    def launch(out):
        handle = chosen[(1, 1)](
            q.to("mps"),
            keys,
            v.to("mps"),
            out,
            *strides,
            *strides,
            *strides,
            *strides,
            1,
            1,
            n,
            BLOCK_M=n,
            BLOCK_N=32,
            HEAD_DIM=dim,
            IS_CAUSAL=False,
        )
        torch.mps.synchronize()
        return handle

    apart = torch.full((1, 1, n, dim), POISON, device="mps")
    handle = launch(apart)
    assert handle.metadata.attention_outputs is None and handle.metadata.flash_attention is None
    assert MMA not in handle.asm["msl"] and TILED not in handle.asm["msl"]
    torch.testing.assert_close(apart.cpu().double(), expected, atol=1e-3, rtol=0)
    assert launched.templates == [] and len(launched.shaders) == 1

    before = storage.clone()
    with pytest.raises(MetalNonRecoverableError, match=r"lane order: .* argument\(s\) K, Out:"):
        launch(keys)
    assert torch.equal(storage, before)
    assert launched.templates == [] and len(launched.shaders) == 1

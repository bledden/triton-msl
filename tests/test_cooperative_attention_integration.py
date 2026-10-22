"""CPU production-path controls for the default-off attention candidate."""

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from tests.test_attention_capability import OPEN_CASES, compile_case
from tests.test_cooperative_attention_candidate import fixture
from triton_msl.autotuning._submission import SubmissionState
from triton_msl.backend import _cooperative_dispatch as dispatch
from triton_msl.backend._attention_variant import validate_descriptor
from triton_msl.backend._launch_contract import seal_launch_metadata, validate_launch_metadata
from triton_msl.backend.compiler import MetalOptions
from triton_msl.codegen._cooperative_attention import build_candidate, make_descriptor
from triton_msl.errors import MetalNonRecoverableError


def descriptor():
    obj, plan = fixture()
    scalar = obj.kb.build()
    return make_descriptor(obj, plan, scalar, build_candidate(obj, plan)), scalar


def metadata(d):
    return dict(
        name=d["primary"],
        execution_contract="controlled",
        num_warps=4,
        num_ctas=1,
        shared=0,
        block_size=32,
        output_arg_indices=[4],
        needs_2d_grid=False,
        mm_two_kernel=None,
        fast_matmul=None,
        quant_matmul=None,
        flash_attention=d,
        batched_dot_bounds=None,
        device_assert=None,
    )


def test_default_off_and_only_proved_tuple_gets_descriptor():
    assert MetalOptions().experimental_cooperative_attention is False
    assert MetalOptions().hash() != MetalOptions(experimental_cooperative_attention=True).hash()
    for case in OPEN_CASES:
        baseline = compile_case(case)
        source = baseline.lower()
        assert baseline._flash_attention is None
        candidate = compile_case(case)
        candidate.options = MetalOptions(experimental_cooperative_attention=True)
        assert candidate.lower() == source
        d = candidate._flash_attention
        if case[1] == 32:
            assert d is not None
            validate_descriptor(d, primary=candidate.kb.name, scalar_source=source)
            assert d["scalar_sha256"] == hashlib.sha256(source.encode()).hexdigest()
            assert (candidate._replay_shared_bytes, candidate._actual_dispatch_threads) == (0, 32)
        else:
            assert d is None


@pytest.mark.parametrize("damage", ["threads", "shared", "entry", "source", "roles", "arg_count", "scalar_threads"])
def test_corrupt_descriptor_refuses(damage):
    d, scalar = descriptor()
    if damage == "entry":
        d[damage] = "other"
    elif damage == "source":
        d[damage] += " altered"
    elif damage == "roles":
        d[damage][0] = d[damage][1]
    else:
        d[damage] = -1
    with pytest.raises(MetalNonRecoverableError):
        validate_descriptor(d, primary="attention", scalar_source=scalar)


def test_real_launch_record_seals_both_sources_and_variant_geometry():
    d, scalar = descriptor()
    m = metadata(d)
    seal_launch_metadata(m)
    snapshot = validate_launch_metadata(m)
    assert json.loads(snapshot)[9] == d
    validate_descriptor(d, primary="attention", scalar_source=scalar, scalar_threads=32)
    for bad_source, bad_size in ((d["source"], 32), (scalar, 512), (scalar + "\n", 32)):
        with pytest.raises(MetalNonRecoverableError, match="binding"):
            validate_descriptor(d, primary="attention", scalar_source=bad_source, scalar_threads=bad_size)
    changed = copy.deepcopy(m)
    changed["flash_attention"]["scalar_sha256"] = "0" * 64
    with pytest.raises(MetalNonRecoverableError, match="producer record"):
        validate_launch_metadata(changed)


def test_scalar_primary_never_claims_sibling_shared_memory():
    d, _ = descriptor()
    m = metadata(d)
    m["shared"] = 16384
    with pytest.raises(MetalNonRecoverableError):
        seal_launch_metadata(m)


def test_layout_reach_alias_and_partial_programs():
    spans = [
        (10000, 32, 33 * 128),
        (20000, 32, 97 * 128),
        (40000, 32, 97 * 128),
        (60000, 64, 33 * 97 * 4),
        (80000, 64, 33 * 256),
    ]
    assert dispatch.layout_eligible(spans, nq=33, nk=97, groups=2)
    assert dispatch.layout_eligible(spans, nq=33, nk=97, groups=1)
    bad = spans.copy()
    bad[4] = (80000, 64, 33 * 256 - 1)
    assert not dispatch.layout_eligible(bad, nq=33, nk=97, groups=2)
    alias = spans.copy()
    alias[4] = (60000, 64, 33 * 256)
    assert dispatch.layout_eligible(alias, nq=33, nk=97, groups=1)
    assert not dispatch.layout_eligible(alias, nq=33, nk=97, groups=2)
    alias[4] = (10000, 32, 33 * 256)
    assert not dispatch.layout_eligible(alias, nq=33, nk=97, groups=1)
    for nq, nk, groups in ((-1, 97, 1), (33, 0, 1), (33, 97, True), (2**30, 97, 1)):
        assert not dispatch.layout_eligible(spans, nq=nq, nk=nk, groups=groups)


def test_distinct_tight_opaque_handles_do_not_imply_overlap():
    # These storage handles/offsets/sizes are from the retained MPS layout diagnosis.
    # Handle spacing (~448 bytes) says nothing about the buffers' physical ranges.
    spans = [
        (32580964736, 32, 2432),
        (32580965184, 32, 12416),
        (32580965632, 32, 12416),
        (32580966080, 64, 7372),
        (32580966528, 64, 8448),
    ]
    assert dispatch.layout_eligible(spans, nq=19, nk=97, groups=1)


@pytest.mark.parametrize("read_role,read_index", [("Q", 0), ("K", 1), ("V", 2)])
def test_same_buffer_disjoint_output_range_is_admitted(read_role, read_index):
    spans = [
        (10000, 32, 33 * 128),
        (20000, 32, 97 * 128),
        (40000, 32, 97 * 128),
        (60000, 64, 33 * 97 * 4),
        (80000, 64, 33 * 256),
    ]
    disjoint = spans.copy()
    read_end = spans[read_index][1] + spans[read_index][2]
    disjoint[4] = (spans[read_index][0], read_end, 33 * 256)
    assert dispatch.layout_eligible(disjoint, nq=33, nk=97, groups=2), read_role


@pytest.mark.parametrize("read_role,read_index", [("K", 1), ("V", 2)])
@pytest.mark.parametrize("groups", [1, 2])
def test_output_overlap_with_key_or_value_keeps_scalar(read_role, read_index, groups):
    spans = [
        (10000, 32, 33 * 128),
        (20000, 32, 97 * 128),
        (40000, 32, 97 * 128),
        (60000, 64, 33 * 97 * 4),
        (80000, 64, 33 * 256),
    ]
    assert dispatch.layout_eligible(spans, nq=33, nk=97, groups=groups)
    alias = spans.copy()
    alias[4] = (spans[read_index][0], spans[read_index][1], 33 * 256)
    assert not dispatch.layout_eligible(alias, nq=33, nk=97, groups=groups), read_role


class Kernel:
    thread_execution_width = 32
    max_threads_per_threadgroup = 1024
    static_thread_group_memory_length = 16384

    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class Runtime:
    def __init__(self, kernel):
        self.kernel = kernel
        self.marked = []
        self.dispatches = 0
        self.library = SimpleNamespace(attention__attncoop16=kernel)

    def is_unsupported(self, source):
        return source in self.marked

    def mark_unsupported(self, source):
        self.marked.append(source)

    def get_library(self, source):
        return self.library

    def dispatch(self, library, entry, args, **kwargs):
        self.dispatches += 1
        getattr(library, entry)(*args, **kwargs)


def run(rt, d, state, **kwargs):
    return dispatch.dispatch_cooperative_attention(
        rt, d, [], (2, 1, 1), device_limits=(1024, 32768), submission_state=state, **kwargs
    )


def test_pipeline_selection_and_actual_dispatch_geometry(monkeypatch):
    monkeypatch.setattr(dispatch, "_tensor_layout", lambda *a: True)
    d, _ = descriptor()
    rt, state = Runtime(Kernel()), SubmissionState()
    assert run(rt, d, state)
    assert rt.dispatches == 1 and state.attempted
    assert rt.kernel.calls == [((), {"threads": 1024, "group_size": 512})]


@pytest.mark.parametrize(
    "attribute,value",
    [("thread_execution_width", 16), ("max_threads_per_threadgroup", 256), ("static_thread_group_memory_length", 0)],
)
def test_unsuitable_pipeline_keeps_scalar_before_invocation(monkeypatch, attribute, value):
    monkeypatch.setattr(dispatch, "_tensor_layout", lambda *a: True)
    d, _ = descriptor()
    kernel = Kernel()
    setattr(kernel, attribute, value)
    rt, state = Runtime(kernel), SubmissionState()
    assert not run(rt, d, state)
    assert not state.attempted and rt.dispatches == 0


def test_compile_failure_does_not_poison_scalar(monkeypatch):
    monkeypatch.setattr(dispatch, "_tensor_layout", lambda *a: True)
    d, scalar = descriptor()
    rt, state = Runtime(Kernel()), SubmissionState()
    monkeypatch.setattr(rt, "get_library", lambda _: (_ for _ in ()).throw(RuntimeError("compile")))
    assert not run(rt, d, state)
    assert rt.marked == [d["source"]] and scalar not in rt.marked and not state.attempted


@pytest.mark.parametrize("where", ["dispatch", "exit_hook"])
def test_no_replay_after_possible_submission(monkeypatch, where):
    monkeypatch.setattr(dispatch, "_tensor_layout", lambda *a: True)
    d, _ = descriptor()
    rt, state = Runtime(Kernel()), SubmissionState()
    error = RuntimeError("after submission")

    def fail(*a, **kw):
        raise error

    kwargs = {}
    if where == "dispatch":
        monkeypatch.setattr(rt, "dispatch", fail)
    else:
        kwargs["launch_exit_hook"] = fail
    with pytest.raises(RuntimeError) as caught:
        run(rt, d, state, **kwargs)
    assert caught.value is error and state.attempted and not rt.marked


def test_dispatch_uses_same_callable_as_pipeline_probe(monkeypatch):
    monkeypatch.setattr(dispatch, "_tensor_layout", lambda *a: True)
    d, _ = descriptor()
    rt = Runtime(Kernel())
    original = rt.kernel

    class SwappingKernel(Kernel):
        @property
        def static_thread_group_memory_length(self):
            rt.library.attention__attncoop16 = Kernel()
            return 16384

    rt.library.attention__attncoop16 = original = SwappingKernel()
    assert run(rt, d, SubmissionState())
    assert len(original.calls) == 1
    assert not rt.library.attention__attncoop16.calls


def test_emitter_retains_public_occupancy_convention(monkeypatch):
    from triton_msl.codegen import mlir_walker, msl_emitter

    graph = compile_case(OPEN_CASES[0]).graph
    monkeypatch.setattr(mlir_walker, "walk_ttgir", lambda *a: graph)
    mod = SimpleNamespace(walk=lambda callback: None)
    ordinary = {"shared": 17}
    candidate = {"shared": 17}
    old_source = msl_emitter.emit_msl(mod, ordinary, MetalOptions())
    new_source = msl_emitter.emit_msl(mod, candidate, MetalOptions(experimental_cooperative_attention=True))
    assert new_source == old_source
    assert ordinary["shared"] == 17  # Unrelated/default-off metadata untouched.
    assert candidate["shared"] == 1 and 32768 // candidate["shared"] == 32768
    assert candidate["flash_attention"]["scalar_shared"] == 0
    assert candidate["flash_attention"]["shared"] == 16384


def launcher_fixture(monkeypatch, stash):
    from triton_msl.backend import compiler, driver, _cache_contract
    from triton_msl.backend._launch_contract import pack_launch_metadata

    d, scalar = descriptor()
    m = metadata(d)
    m["msl_hash"] = "a" * 64
    seal_launch_metadata(m)
    names = ["Q", "K", "V", "Bias", "Out", "NQ", "NK"]
    src = SimpleNamespace(
        constants={},
        fn=SimpleNamespace(arg_names=names),
        signature=dict(zip(names, ["*fp16"] * 3 + ["*fp32"] * 2 + ["i32"] * 2)),
    )
    monkeypatch.setattr(_cache_contract, "validate_execution_contract", lambda x: x)
    monkeypatch.setattr(_cache_contract, "install_jit_policy_guard", lambda *a: None)
    monkeypatch.setattr(compiler, "_load_stashed_msl", lambda key: stash(d, scalar))
    return driver.MetalLauncher(src, SimpleNamespace(**m)), pack_launch_metadata(m)


@pytest.mark.parametrize("damage", ["wrong_source", "wrong_geometry", "sibling_as_scalar"])
def test_launcher_refuses_stale_stash_before_runtime(monkeypatch, damage):
    def stash(d, scalar):
        return {
            "wrong_source": (scalar + "\n", 32),
            "wrong_geometry": (scalar, 512),
            "sibling_as_scalar": (d["source"], 32),
        }[damage]

    with pytest.raises(MetalNonRecoverableError, match="binding"):
        launcher_fixture(monkeypatch, stash)


@pytest.mark.parametrize("present", [False, True])
def test_live_launcher_requires_scalar_stash_before_candidate(monkeypatch, present):
    import torch
    from triton_msl.backend import driver, _launch_signature, _host_dispatch

    launcher, packed = launcher_fixture(monkeypatch, lambda d, scalar: (scalar, 32) if present else None)
    events = []

    class Tensor:
        device = "mps"

        def data_ptr(self):
            return 4096

    class ScalarFallback(Exception):
        pass

    class Utils:
        device = SimpleNamespace(
            maxThreadsPerThreadgroup=lambda: SimpleNamespace(width=1024), maxThreadgroupMemoryLength=lambda: 32768
        )

        @property
        def buffer_pool(self):
            raise ScalarFallback

    args = [Tensor() for _ in range(5)] + [19, 97]
    monkeypatch.setattr(torch, "Tensor", Tensor)
    monkeypatch.setattr(_launch_signature, "bind_arguments_with_plan", lambda *a: (args, [], [], []))
    monkeypatch.setattr(driver, "_get_utils", lambda: Utils())
    monkeypatch.setattr(driver, "_get_compile_shader_runtime", lambda: SimpleNamespace(available=lambda: True))
    monkeypatch.setattr(_host_dispatch, "check_pending", lambda: None)
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "1")
    monkeypatch.setenv("TRITON_MSL_FA_FAST", "1")
    monkeypatch.setattr(
        dispatch, "dispatch_cooperative_attention", lambda *a, **kw: events.append("cooperative") or True
    )
    if present:
        launcher(1, 1, 1, None, None, packed, None, None, None, *args)
        assert events == ["cooperative"]
    else:
        with pytest.raises(ScalarFallback):
            launcher(1, 1, 1, None, None, packed, None, None, None, *args)
        assert events == []

"""Resident/restored handles must check current semantics BEFORE side effects.

Controlled CPU launcher witnesses, not a GPU warm-upgrade or packed-ABI proof.
"""

import json
from types import SimpleNamespace

import pytest
import triton  # noqa: F401
from triton.backends.compiler import GPUTarget

from triton_msl.backend import _cache_contract as contract, compiler, driver
from triton_msl.errors import MetalNonRecoverableError


def _stamp():
    # Independent literal schema oracle also works before the producer exists.
    return json.dumps(
        {"schema": 1, "source": contract.source_contract(), "toolchain": contract.toolchain_identity()},
        sort_keys=True,
        separators=(",", ":"),
    )


def test_stamp_text_reuse_never_reuses_live_validation(monkeypatch):
    source = {
        "schema": 3,
        "implementation": "implementation",
        "frameworks": "framework",
        "label": "label",
        "policy": {key: False for key in contract._STAMP_FLAGS},
    }
    source["policy"]["CPP_SKIP"] = ""
    source["policy"]["HOST_EXTENT"] = False
    calls = []

    def current():
        calls.append("source")
        return source

    def toolchain():
        calls.append("toolchain")
        return "toolchain"

    monkeypatch.setattr(contract, "source_contract", current)
    monkeypatch.setattr(contract, "toolchain_identity", toolchain)
    contract._encode_known_stamp.cache_clear()
    first = contract.execution_contract()
    assert contract.execution_contract() == first
    assert calls == ["source", "toolchain", "source", "toolchain"]
    assert contract._encode_known_stamp.cache_info().hits == 1
    # Same dict, changed primitive or schema, and Python-equal JSON type aliases.
    for key, value in [
        *((key, True) for key in contract._STAMP_FLAGS),
        ("CPP_SKIP", "changed"),
        ("HOST_EXTENT", True),
        ("HOST_EXTENT", "unavailable"),
        ("HOST_EXTENT", 0),
        ("FA_FAST", 0),
        ("FA_FAST", 1.0),
    ]:
        prior = source["policy"][key]
        source["policy"][key] = value
        assert contract.execution_contract() == _stamp()
        assert contract.execution_contract() != first
        source["policy"][key] = prior
        assert contract.execution_contract() == first
    source["schema"] = True
    source["extra"] = ["live"]
    assert contract.execution_contract() == _stamp()
    before = contract.execution_contract()
    source["extra"][0] = "changed"
    assert contract.execution_contract() != before
    assert contract.execution_contract() == _stamp()

    def failed_check():
        raise MetalNonRecoverableError("current native identity failed")

    monkeypatch.setattr(contract, "toolchain_identity", failed_check)
    with pytest.raises(MetalNonRecoverableError, match="current native identity failed"):
        contract.execution_contract()


@pytest.fixture(autouse=True)
def controlled_dependencies(monkeypatch, tmp_path):
    monkeypatch.setattr(contract, "toolchain_identity", lambda: "controlled-toolchain")
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path))


def _metadata(stamp):
    data = dict(
        name="execution_contract",
        num_warps=4,
        num_ctas=1,
        shared=0,
        block_size=128,
        execution_contract=stamp,
        output_arg_indices=None,
        needs_2d_grid=False,
        mm_two_kernel=None,
        fast_matmul=None,
        quant_matmul=None,
        flash_attention=None,
        batched_dot_bounds=None,
        device_assert=None,
    )
    fields = (
        "num_warps",
        "num_ctas",
        "shared",
        "block_size",
        "output_arg_indices",
        "needs_2d_grid",
        "mm_two_kernel",
        "fast_matmul",
        "quant_matmul",
        "flash_attention",
        "batched_dot_bounds",
        "device_assert",
    )
    data["launch_contract"] = json.dumps(
        {
            "schema": 2,
            "kind": "metal-packed-launch",
            "name": data["name"],
            "execution_contract": stamp,
            "fields": {f: data[f] for f in fields},
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return SimpleNamespace(**data)


def _launcher(stamp):
    source = SimpleNamespace(constants={}, signature={}, fn=SimpleNamespace(arg_names=[]))
    return driver.MetalLauncher(source, _metadata(stamp))


@pytest.mark.parametrize("value", [None, "", "{}", '{"schema":true}'])
def test_unbound_restored_launcher_refuses_before_stash(value, monkeypatch):
    def forbidden(*args):
        pytest.fail("unbound launcher reached a cache lookup")

    monkeypatch.setattr(compiler, "_load_stashed_msl", forbidden)
    with pytest.raises(MetalNonRecoverableError, match="execution contract"):
        _launcher(value)


@pytest.mark.parametrize(
    "name,a,b",
    [
        ("MEPT", "1", "0"),
        ("QUANT_MATMUL", "1", "0"),
        ("FAST_MATMUL", "1", "0"),
        ("COMPILE_SHADER", "1", "0"),
        ("FA_FAST", "1", "0"),
        ("INFER_LAYOUT", "1", "0"),
        ("LEGACY", "1", "0"),
        ("USE_CPP", "0", "1"),
        ("FORCE_PYTHON", "0", "1"),
        ("FA_HALF_ACCUM", "0", "1"),
        ("CPP_SKIP", "", "reduce"),
    ],
)
def test_resident_policy_change_refuses_before_hooks_or_runtime(monkeypatch, name, a, b):
    monkeypatch.setenv("TRITON_MSL_" + name, a)
    launcher = _launcher(_stamp())
    monkeypatch.setenv("TRITON_MSL_" + name, b)
    events = []

    def runtime():
        events.append("runtime")
        pytest.fail("stale launcher reached the runtime")

    monkeypatch.setattr(driver, "_get_utils", runtime)
    with pytest.raises(MetalNonRecoverableError, match="execution contract"):
        launcher(1, 1, 1, None, None, None, None, lambda _: events.append("enter"), None)
    assert events == []


@pytest.mark.parametrize("dependency", ["implementation_identity", "toolchain_identity"])
def test_restored_handle_cannot_recertify_old_dependency_bytes(dependency, monkeypatch):
    original = _stamp()
    monkeypatch.setattr(contract, dependency, lambda: "changed-implementation-without-label-bump")
    with pytest.raises(MetalNonRecoverableError, match="execution contract"):
        _launcher(original)


def test_current_handle_reaches_existing_launch_boundary(monkeypatch):
    # Production constructs the backend (and its lazy module map) before a
    # producer captures identity. A launcher stamped before that initialization
    # is genuinely stale and must not be accepted just to reach this sentinel.
    backend = compiler.MetalBackend(GPUTarget("metal", "apple-m4", 32))
    launcher = _launcher(_stamp())

    class ReachedRuntime(Exception):
        pass

    events = []

    def runtime():
        raise ReachedRuntime

    monkeypatch.setattr(driver, "_get_utils", runtime)
    with pytest.raises(ReachedRuntime):
        packed = backend.pack_metadata(_metadata(_stamp()))
        launcher(1, 1, 1, None, None, packed, None, lambda _: events.append("enter"), None)
    assert events == ["enter"]


def test_stale_packed_metadata_refuses_before_defaults_are_invented(monkeypatch):
    monkeypatch.setenv("TRITON_MSL_INFER_LAYOUT", "0")
    metadata = _metadata(_stamp())
    monkeypatch.setenv("TRITON_MSL_INFER_LAYOUT", "1")
    with pytest.raises(MetalNonRecoverableError, match="execution contract"):
        compiler.MetalBackend(GPUTarget("metal", "apple-m4", 32)).pack_metadata(metadata)


@pytest.mark.parametrize("change", ["none", "before", "during", "native-before", "native-during"])
def test_only_successful_unchanged_compile_produces_execution_stamp(monkeypatch, change):
    monkeypatch.setenv("TRITON_MSL_INFER_LAYOUT", "0")
    monkeypatch.setenv("TRITON_MSL_USE_CPP", "0")
    backend = compiler.MetalBackend(GPUTarget("metal", "apple-m4", 32))
    native = ["original-loaded-order"]
    if change.startswith("native-"):
        monkeypatch.setattr(contract, "framework_identity", lambda: native[0])
    calls = []

    def produce(source, metadata, options):
        calls.append("compile")
        if change == "during":
            monkeypatch.setenv("TRITON_MSL_INFER_LAYOUT", "1")
        if change == "native-during":
            native[0] = "original-loaded-order-plus-interposed-provider"
        return b"controlled fresh binary"

    monkeypatch.setattr(backend, "make_metallib", produce)
    stages = {}
    backend.add_stages(stages, compiler.MetalOptions())
    metadata = {"name": "execution_contract", "num_warps": 4, "num_ctas": 1}
    if change == "before":
        monkeypatch.setenv("TRITON_MSL_INFER_LAYOUT", "1")
    if change == "native-before":
        native[0] = "original-loaded-order-plus-interposed-provider"
    if change == "none":
        assert stages["metallib"]("source", metadata) == b"controlled fresh binary"
        assert metadata["execution_contract"] == _stamp()
    else:
        with pytest.raises(MetalNonRecoverableError, match="execution contract"):
            stages["metallib"]("source", metadata)
        assert "execution_contract" not in metadata
        assert calls == ([] if change in ("before", "native-before") else ["compile"])


class _HookableJit:
    def __init__(self):
        self.arg_names = []
        self.device_caches = {"device": "controlled resident kernel/binder caches"}
        self.pre_run_hooks = []

    def add_pre_run_hook(self, hook):
        self.pre_run_hooks.append(hook)


def test_per_jit_guard_preserves_user_hooks_and_only_clears_its_owner(monkeypatch):
    monkeypatch.setenv("TRITON_MSL_INFER_LAYOUT", "1")
    owner, unrelated = _HookableJit(), _HookableJit()
    events = []
    owner.add_pre_run_hook(lambda *a, **kw: events.append((a, kw)))
    source = SimpleNamespace(constants={}, signature={}, fn=owner)
    driver.MetalLauncher(source, _metadata(_stamp()))
    driver.MetalLauncher(source, _metadata(_stamp()))
    assert len(owner.pre_run_hooks) == 2  # one user hook + ONE backend guard
    for hook in owner.pre_run_hooks:
        hook(5, parameter=7)
    assert events == [((5,), {"parameter": 7})]
    assert owner.device_caches and unrelated.device_caches
    monkeypatch.setenv("TRITON_MSL_INFER_LAYOUT", "0")
    for hook in owner.pre_run_hooks:
        hook(6, parameter=8)
    assert events[-1] == ((6,), {"parameter": 8})
    assert owner.device_caches == {} and unrelated.device_caches
    owner.device_caches["new"] = "recompiled under current policy"
    owner.pre_run_hooks[-1]()
    assert owner.device_caches == {"new": "recompiled under current policy"}


def test_guard_does_not_keep_a_jit_function_alive():
    import gc
    import weakref

    owner = _HookableJit()
    ref = weakref.ref(owner)
    assert contract.install_jit_policy_guard(owner, _stamp())
    (guard,) = owner.pre_run_hooks
    del owner
    gc.collect()
    assert ref() is None
    guard()  # harmless after owner is gone

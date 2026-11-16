"""CPU proof controls for geometry/ABI; device labels are explicit doubles."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch

from triton_msl.backend._generic_launch import generic_2d_geometry, host_launch_geometry


@pytest.fixture
def case(monkeypatch):
    launcher = SimpleNamespace(
        _msl="arbitrary source already validated by stash",
        _msl_block_size=64,
        _execution_contract=json.dumps({"source": {"policy": {"USE_CPP": False}}}),
    )
    arguments = [torch.zeros(3 * 5 * 64), torch.zeros(3 * 5 * 64), 960]
    metadata = [4, 1, 0, 64, [1], True, None, None, None, None, None, None]
    monkeypatch.setattr(torch.Tensor, "device", property(lambda self: torch.device("mps")))
    return launcher, arguments, ["*fp32", "*fp32", "i32"], metadata, (3, 5, 1)


def test_host_mapping_is_preserved_including_existing_clamp():
    assert host_launch_geometry((3, 5, 7), 64, True) == ((3, 5, 7), (64, 1, 1))
    assert host_launch_geometry((3, 5, 7), 64, False) == ((105, 1, 1), (64, 1, 1))
    assert host_launch_geometry((3, 5, 1), 2048, True) == ((3, 5, 1), (1024, 1, 1))
    assert host_launch_geometry((0, 5, 1), 64, True) == ((0, 5, 1), (64, 1, 1))


@pytest.mark.parametrize("group", [32, 64, 256, 1024])
def test_nonsquare_multiy_geometry_matches_host_without_clamp(case, group):
    launcher, args, sigs, meta, grid = case
    launcher._msl_block_size = meta[3] = group
    threads, tg = generic_2d_geometry(*case)
    host_grid, host_tg = host_launch_geometry(grid, group, True)
    assert tg == host_tg and threads == tuple(a * b for a, b in zip(host_grid, host_tg))
    assert threads == (3 * group, 5, 1)


def test_actual_views_and_aliases_are_not_rebased_or_deduplicated(case):
    launcher, args, sigs, meta, grid = case
    owner = torch.zeros(3000)
    args[:2] = [owner[16:976], owner[1100:2060]]
    first, second = args[:2]
    assert generic_2d_geometry(*case) == ((192, 5, 1), (64, 1, 1))
    assert args[0] is first and args[1] is second
    assert first.storage_offset() == 16 and second.storage_offset() == 1100
    args[1] = first
    assert generic_2d_geometry(*case) == ((192, 5, 1), (64, 1, 1))


@pytest.mark.parametrize(
    "kind",
    [
        "group_mismatch",
        "group_clamp",
        "no_stash_group",
        "template",
        "packing",
        "z",
        "zero_grid",
        "float_grid",
        "overflow_grid",
        "flat_grid",
        "cpp",
        "unknown_policy",
        "dtype",
        "subclass",
        "noncontiguous",
        "empty",
        "half_scalar",
        "high_u64",
        "aggregate",
    ],
)
def test_unproved_forms_retain_host_eligibility(case, kind):
    launcher, args, sigs, meta, grid = case
    if kind == "group_mismatch":
        launcher._msl_block_size = 32
    elif kind == "group_clamp":
        launcher._msl_block_size = meta[3] = 2048
    elif kind == "no_stash_group":
        launcher._msl_block_size = None
    elif kind == "template":
        meta[6] = {"special": True}
    elif kind == "packing":
        args *= 11
        sigs *= 11
    elif kind == "z":
        grid = (3, 5, 2)
    elif kind == "zero_grid":
        grid = (0, 5, 1)
    elif kind == "float_grid":
        grid = (3.0, 5, 1)
    elif kind == "overflow_grid":
        grid = (1 << 30, 5, 1)
    elif kind == "flat_grid":
        meta[5] = False
    elif kind == "cpp":
        launcher._execution_contract = json.dumps({"source": {"policy": {"USE_CPP": True}}})
    elif kind == "unknown_policy":
        launcher._execution_contract = "{}"
    elif kind == "dtype":
        args[0] = torch.zeros(960, dtype=torch.int32)
    elif kind == "subclass":
        args[0] = torch.nn.Parameter(args[0])
    elif kind == "noncontiguous":
        args[0] = torch.zeros(1920)[::2]
    elif kind == "empty":
        args[0] = torch.zeros(0)
    elif kind == "half_scalar":
        sigs[2] = "fp16"
        args[2] = 1.0
    elif kind == "high_u64":
        sigs[2] = "u64"
        args[2] = 1 << 63
    elif kind == "aggregate":
        sigs[2] = ("i32", "i32")
        args[2] = (1, 2)
    assert generic_2d_geometry(launcher, args, sigs, meta, grid) is None


@pytest.mark.parametrize(
    "declared,value",
    [
        ("i8", -128),
        ("u8", 255),
        ("i16", -32768),
        ("u16", 65535),
        ("i32", -(1 << 31)),
        ("u32", (1 << 32) - 1),
        ("i64", 1 << 40),
        ("i64", -(1 << 63)),
        ("u64", (1 << 63) - 1),
        ("fp32", 1.25),
    ],
)
def test_scalar_checked_bytes_are_representable_by_raw_bridge(case, declared, value):
    from triton_msl.backend._launch_signature import scalar_bytes

    launcher, args, sigs, meta, grid = case
    sigs[2] = declared
    args[2] = value
    payload = scalar_bytes(value, declared)
    import struct

    raw = struct.pack("<f", value) if declared == "fp32" else struct.pack("<q", value)
    assert raw[: len(payload)] == payload
    assert generic_2d_geometry(*case) == ((192, 5, 1), (64, 1, 1))


@pytest.mark.parametrize("mode", ["clean", "assertion", "dispatch_error", "exit_error", "compile_miss", "bad_scalar"])
@pytest.mark.parametrize("with_tail_bounds", [False, True])
def test_public_launcher_keeps_validation_and_no_replay(case, monkeypatch, mode, with_tail_bounds):
    import triton.backends
    from triton_msl.backend import driver, _cache_contract, _launch_contract, _launch_signature
    from triton_msl.errors import PostSubmitError, MetalDeviceAssertionError, MetalNonRecoverableError

    stub, args, sigs, meta, grid = case
    meta[11] = {"schema": 1, "messages": ["controlled assertion"], "buffer_index": 3}
    if with_tail_bounds:
        meta[10] = [
            "matmul_full_k_storage_v1",
            32,
            [0, 1, 1],
            [["literal", 32, "i32"], ["literal", 32, "i32"], ["arg", 2, "i32"]],
            [["literal", 1, "i64"]] * 6,
            [["i32", "i32", False]] * 3,
        ]
    events = []
    flags = []

    class HostFallback(Exception):
        pass

    class Utils:
        @property
        def buffer_pool(self):
            events.append("host")
            raise HostFallback

    class Runtime:
        def available(self):
            return True

        def is_unsupported(self, s):
            return False

        def mark_unsupported(self, s):
            events.append("blacklist")

        def get_library(self, s):
            if mode == "compile_miss":
                raise RuntimeError("pre invocation")
            return object()

        def dispatch(self, lib, name, actual, **geometry):
            events.append("dispatch")
            flags.append(actual[-1])
            assert all(a is b for a, b in zip(actual[:3], args))
            assert geometry == dict(threads=(192, 5, 1), group_size=(64, 1, 1))
            assert actual[-1].item() == 0
            if mode == "assertion":
                actual[-1].fill_(1)
            if mode == "dispatch_error":
                raise RuntimeError("attempted")

    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "1")
    monkeypatch.setattr(_cache_contract, "execution_contract", lambda: stub._execution_contract)
    monkeypatch.setattr(driver, "_get_utils", lambda: Utils())
    monkeypatch.setattr(driver, "_get_compile_shader_runtime", lambda: Runtime())
    original_zeros = torch.zeros

    def zeros(*a, **kw):
        if kw.get("device") == "mps":
            kw["device"] = "cpu"
        return original_zeros(*a, **kw)

    monkeypatch.setattr(torch, "zeros", zeros)
    for obj, name in (
        (_cache_contract, "validate_execution_contract"),
        (_launch_contract, "validate_packed_launch"),
        (_launch_signature, "bind_arguments_with_plan"),
        (driver, "_matmul_tail_bounds_reason"),
    ):
        original = getattr(obj, name)

        def record(*a, _fn=original, _name=name, **kw):
            events.append(_name)
            return _fn(*a, **kw)

        monkeypatch.setattr(obj, name, record)
    launcher = driver.MetalLauncher.__new__(driver.MetalLauncher)
    launcher.__dict__.update(vars(stub))
    launcher.arg_names = ["a", "b", "n"]
    launcher.signature = dict(zip(launcher.arg_names, sigs))
    launcher.constexpr_indices = set()
    launcher._binding_plan = _launch_signature.make_binding_plan(launcher.arg_names, launcher.signature)
    launcher._packed_contract = _launch_contract._canonical(meta)
    launcher.kernel_name = "unrecognized_generic"

    def leave(_):
        if mode == "exit_error":
            raise RuntimeError("exit hook")

    if mode == "bad_scalar":
        args[2] = 1 << 40

    def call():
        launcher(*grid, None, None, meta, None, None, leave, *args)

    if mode == "clean":
        call()
        call()
        assert flags[0] is not flags[1]
    else:
        error = {
            "assertion": MetalDeviceAssertionError,
            "dispatch_error": PostSubmitError,
            "exit_error": PostSubmitError,
            "compile_miss": HostFallback,
            "bad_scalar": MetalNonRecoverableError,
        }[mode]
        with pytest.raises(error):
            call()
        if mode in ("assertion", "dispatch_error", "exit_error"):
            assert events.count("dispatch") == 1 and "host" not in events and "blacklist" not in events
        else:
            assert "dispatch" not in events
    assert events[:3] == ["validate_execution_contract", "validate_packed_launch", "bind_arguments_with_plan"]
    if with_tail_bounds and mode != "bad_scalar":
        assert events[3] == "_matmul_tail_bounds_reason"
        assert events.count("_matmul_tail_bounds_reason") == (2 if mode == "clean" else 1)

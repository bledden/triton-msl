"""Submission is a monotonic boundary, not an inference from an exception message."""

import sys
import pytest
import torch
import triton.backends  # backend discovery before direct driver imports
from triton_msl.backend.compile_shader_runtime import CompileShaderRuntime
from triton_msl.autotuning._fa_dispatch import dispatch_flash_attention
from triton_msl.autotuning._fast_matmul_dispatch import dispatch_fast_matmul
from triton_msl.autotuning._quant_matmul_dispatch import dispatch_quant_matmul
from triton_msl.errors import PostSubmitError
from triton_msl.autotuning._submission import SubmissionState


class _Library:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def kernel(*args, **kwargs):
            self.calls.append(name)

        return kernel


def _case(family, monkeypatch):
    # CPU buffers/metadata only. The library call below records effects; no GPU work.
    # The output is its own buffer: a replacement declines an output that overlaps an operand.
    t, out = torch.zeros(4096), torch.zeros(4096)
    if family == "mla":

        class View:
            device = "mps"

            def __init__(self):
                self.tensor = torch.ones(1, dtype=torch.float32)

            def __getattr__(self, name):
                return getattr(self.tensor, name)

        real_view = torch.as_strided
        monkeypatch.setattr(torch, "as_strided", lambda value, *a, **k: real_view(value.tensor, *a, **k))
        refs = {role: ("c1",) * 4 for role in ("q", "q_rope", "k", "k_rope", "v", "out")}
        desc = ("mla", "source", "mla", 32, *range(6), "c1", "c1", "c1", 32, (1, 1, 1), refs, ("f32",) * 6)
        return lambda rt, hook, **kw: dispatch_flash_attention(
            rt, desc, "mla", [View() for _ in range(6)], 1, 1, 1, launch_exit_hook=hook, **kw
        )
    if family == "fa":
        return lambda rt, hook, **kw: dispatch_flash_attention(
            rt, ("flash_attention", "source", 32), "fa", [], 1, 1, 1, launch_exit_hook=hook, **kw
        )
    if family == "matmul":
        desc = ("source", 3, 4, 5, 32, 128, None, None, (), ("2d", 32, 128, True, True))
        return lambda rt, hook, **kw: dispatch_fast_matmul(
            rt, desc, [t, t, out, 32, 128, 32], grid=(1, 1, 1), launch_exit_hook=hook, **kw
        )
    if family == "splitk":
        # Uses actual split-K allocation on CPU and both real helper dispatch sites.
        monkeypatch.setenv("TRITON_MSL_MATMUL_AUTOTUNE", "0")
        desc = ("source", 3, 4, 5, 32, 32, "float", "float", (), ("2d", 32, 32, True, True))
        return lambda rt, hook, **kw: dispatch_fast_matmul(
            rt, desc, [t, t, out, 32, 32, 2048], grid=(1, 1, 1), launch_exit_hook=hook, **kw
        )
    if family == "int8":
        desc = ("source", 5, 6, 7, 32, 32, (), (32, 32))
        return lambda rt, hook, **kw: dispatch_quant_matmul(
            rt, desc, [t, t, out, t, t, 32, 32, 32], grid=(1, 1, 1), launch_exit_hook=hook, **kw
        )
    if family == "gemv":
        desc = ("gemv", "source", 0, 1, 2, 3, 4, 5, 6, (), 32)
        return lambda rt, hook, **kw: dispatch_quant_matmul(
            rt, desc, [t, t, out, t, t, 32, 32], grid=(1, 1, 1), launch_exit_hook=hook, **kw
        )
    if family == "int4_gemv":
        desc = ("gemv_int4", "source", 0, 1, 2, 3, 4, 5, 6, None, None, None, 32, 32)
        return lambda rt, hook, **kw: dispatch_quant_matmul(
            rt, desc, [t, t, out, t, t, 32, 32], grid=(1, 1, 1), launch_exit_hook=hook, **kw
        )
    if family.startswith("pergroup"):
        tag = "pergroup_int4" if "int4" in family else "pergroup_int8"
        # Real CPU tensors: the route reads each tensor's storage before it launches, and an
        # argument whose storage cannot be read is refused.
        buffers = [t, t, out, t, t]
        desc = (
            tag,
            "scalar",
            5,
            6,
            7,
            tuple(range(8, 18)),
            "fast" if family.endswith("fast") else None,
            4,
            4,
            32,
            32,
            (32, 32),
        )
        return lambda rt, hook, **kw: dispatch_quant_matmul(
            rt,
            desc,
            buffers + [32, 32, 32, 32, 1, 32, 1, 32, 1, 32, 1, 32, 1],
            grid=(1, 1, 1),
            launch_exit_hook=hook,
            **kw,
        )
    if family == "symmetric":
        desc = ("sym_int8", "source", 4, 5, 6, (7, -1, 8, -1, 9, -1, -1), (32, 32))
        return lambda rt, hook, **kw: dispatch_quant_matmul(
            rt,
            desc,
            [t, t, out, t, 32, 32, 32, 32, 32, 32],
            grid=(1, 1, 1),
            launch_exit_hook=hook,
            **kw,
        )
    raise AssertionError(family)


FAMILIES = [
    "mla",
    "fa",
    "matmul",
    "splitk",
    "int8",
    "gemv",
    "int4_gemv",
    "pergroup_int8_fast",
    "pergroup_int4_fast",
    "pergroup_scalar",
    "symmetric",
]


@pytest.mark.parametrize("family", FAMILIES)
def test_each_dispatch_family_has_a_proven_submission_boundary(family, monkeypatch):
    call = _case(family, monkeypatch)
    # One node per family; modes are complementary contract checks, not a shape cross-product.
    for mode in ("clean", "compile_miss", "dispatch_after", "dispatch_before", "hook"):
        lib = _Library()
        rt = CompileShaderRuntime()
        state = SubmissionState()

        def get_library(source):
            if mode == "compile_miss":
                raise RuntimeError("known to occur before invocation in this witness")
            return lib

        rt.get_library = get_library

        def hook(meta):
            raise RuntimeError("hook after caller workload")

        def trace(frame, event, arg):
            if frame.f_code is CompileShaderRuntime.dispatch.__code__:
                if mode == "dispatch_before" and event == "call":
                    raise RuntimeError("invocation boundary: cannot assume enqueue state")
                if mode == "dispatch_after" and event == "return":
                    raise RuntimeError("trace callback after real library call")
            return trace

        old_trace = sys.gettrace()
        try:
            if mode.startswith("dispatch_"):
                sys.settrace(trace)
            if mode in ("dispatch_before", "dispatch_after", "hook"):
                with pytest.raises(PostSubmitError):
                    call(rt, hook if mode == "hook" else None, submission_state=state)
                assert state.attempted
                assert len(lib.calls) == (
                    0 if mode == "dispatch_before" else 2 if family == "splitk" and mode == "hook" else 1
                ), (family, mode, lib.calls)
                assert not rt._unsupported, (family, mode, "attempted workload was blacklisted")
            else:
                result = call(rt, None, submission_state=state)
                assert state.attempted is (mode == "clean")
                assert result is (mode == "clean"), (family, mode, result)
                assert len(lib.calls) == (2 if family == "splitk" else 1) if mode == "clean" else not lib.calls
        finally:
            sys.settrace(old_trace)


def test_splitk_second_invocation_cannot_fall_back(monkeypatch):
    call = _case("splitk", monkeypatch)
    lib = _Library()
    rt = CompileShaderRuntime()
    rt.get_library = lambda source: lib
    count = 0

    def trace(frame, event, arg):
        nonlocal count
        if frame.f_code is CompileShaderRuntime.dispatch.__code__ and event == "call":
            count += 1
            if count == 2:
                raise RuntimeError("second invocation; partials already submitted")
        return trace

    old = sys.gettrace()
    try:
        sys.settrace(trace)
        with pytest.raises(PostSubmitError):
            call(rt, None)
    finally:
        sys.settrace(old)
    assert lib.calls == ["mm_sk_partial"]
    assert not rt._unsupported


def test_nested_return_preserves_the_callers_submission_state(monkeypatch):
    from triton_msl.autotuning import _fa_dispatch, _fast_matmul_dispatch, _quant_matmul_dispatch

    children = {
        "mla": _fa_dispatch._dispatch_mla,
        "splitk": _fast_matmul_dispatch._maybe_splitk_dispatch,
        "pergroup_int8_fast": _quant_matmul_dispatch._dispatch_pergroup_int8,
    }
    for family, child in children.items():
        with monkeypatch.context() as patch:
            call = _case(family, patch)
            lib = _Library()
            rt = CompileShaderRuntime()
            rt.get_library = lambda source: lib
            state = SubmissionState()

            def trace(frame, event, arg):
                if frame.f_code is child.__code__ and event == "return" and arg is True:
                    raise RuntimeError("completed nested helper; do not lose submission state")
                return trace

            old = sys.gettrace()
            try:
                sys.settrace(trace)
                with pytest.raises(RuntimeError):
                    call(rt, None, submission_state=state)
            finally:
                sys.settrace(old)
            assert state.attempted, family
            assert len(lib.calls) == (2 if family == "splitk" else 1)
            assert not rt._unsupported
            # The driver's guard must still refuse after unwinding that helper.
            with pytest.raises(PostSubmitError):
                state.reraise_if_attempted(RuntimeError("outer boundary"))


def test_quant_fast_library_miss_still_uses_scalar_once(monkeypatch):
    call = _case("pergroup_int8_fast", monkeypatch)
    lib = _Library()
    rt = CompileShaderRuntime()
    requested = []

    def get_library(source):
        requested.append(source)
        if source == "fast":
            raise RuntimeError("compile miss before any caller-workload invocation")
        return lib

    rt.get_library = get_library
    state = SubmissionState()
    assert call(rt, None, submission_state=state) is True
    assert requested == ["fast", "scalar"]
    assert lib.calls == ["int8_matmul_pergroup"]
    assert state.attempted
    assert rt._unsupported == {"fast"}

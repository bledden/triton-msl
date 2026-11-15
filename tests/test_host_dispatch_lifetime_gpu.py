"""Real Metal submission, controlled host exception, no deliberate device fault."""

import ctypes
import gc
import struct
from types import SimpleNamespace
import weakref

import pytest
import torch
import triton
import triton.language as tl

from triton_msl.backend import _cache_contract, _host_dispatch as lifetime, driver
from triton_msl.errors import PostSubmitError


@triton.jit
def increment(X, O):
    tl.atomic_add(O, tl.load(X))


@pytest.mark.parametrize("host_extent", ["0", "1"])
@pytest.mark.parametrize("boundary", ["commit", "wait"])
def test_submitted_failure_keeps_borrowed_and_pooled_owners_without_replay(monkeypatch, boundary, host_extent):
    if host_extent == "1" and _cache_contract._validation_native is None:
        pytest.skip("Requires the selected compiled native helper")
    assert torch.backends.mps.is_available() and (host_extent == "0" or _cache_contract._HOST_EXTENT_READY)
    assert lifetime.reap_completed() == 0 and not lifetime._retained
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "0")
    monkeypatch.setenv("TRITON_MSL_HOST_EXTENT", host_extent)
    warm_x, warm_o = torch.ones(17), torch.zeros(1048832)
    increment[(1,)](warm_x, warm_o)
    assert warm_o[0].item() == 1
    utils = driver._get_utils()
    queue = utils.command_queue
    failure = RuntimeError("controlled exception after submission")
    commands, commits, refs, mappings = [], [], {}, []
    blocked = True

    class Command:
        def __init__(self, real):
            self.real = real

        def __getattr__(self, name):
            return getattr(self.real, name)

        def commit(self):
            self.real.commit()
            commits.append(True)
            if boundary == "commit":
                raise failure

        def waitUntilCompleted(self):
            raise failure

        def status(self):
            if blocked:
                raise RuntimeError("completion observation deliberately unavailable")
            return self.real.status()

    def create():
        command = Command(queue.commandBuffer())
        commands.append(command)
        return command

    monkeypatch.setattr(utils, "_command_queue", SimpleNamespace(commandBuffer=create))
    acquire = utils.buffer_pool.acquire

    def take(size):
        result = acquire(size)
        mappings.append(weakref.ref(result[1]))
        return result

    monkeypatch.setattr(utils.buffer_pool, "acquire", take)

    def invoke():
        x, out = torch.ones(17), torch.zeros(1048832)
        refs.update(x=weakref.ref(x), out=weakref.ref(out), pointer=out.data_ptr())
        increment[(1,)](x, out)

    try:
        try:
            invoke()
        except RuntimeError as error:
            assert error is failure
            failure.__traceback__ = None
        else:
            pytest.fail("missing injected failure")
        gc.collect()
        assert refs["x"]() is not None and refs["out"]() is not None
        assert mappings and all(ref() is not None and not ref().closed for ref in mappings)
        assert len(lifetime._retained) == 1 and lifetime.reap_completed() == 1
        with pytest.raises(PostSubmitError, match="unconfirmed completion"):
            increment[(1,)](warm_x, warm_o)
        assert len(commands) == len(commits) == 1
        commands[0].real.waitUntilCompleted()
        assert ctypes.cast(refs["pointer"], ctypes.POINTER(ctypes.c_float))[0] == (1.0 if host_extent == "1" else 0.0)
        assert (
            struct.unpack("f", next(iter(lifetime._retained.values())).buffers[1][0].contents().as_buffer(4))[0] == 1.0
        )
        assert lifetime.reap_completed() == 1  # Even a completed command needs an observable witness.
        blocked = False
        assert lifetime.reap_completed() == 0
        gc.collect()
        assert refs["x"]() is None and refs["out"]() is None
        assert all(ref() is None or ref().closed for ref in mappings)
        assert len(commits) == 1
    finally:
        for command in commands:
            command.real.waitUntilCompleted()
        blocked = False
        lifetime.reap_completed()
        failure.__traceback__ = None

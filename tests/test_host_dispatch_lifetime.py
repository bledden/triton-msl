"""Actual runtime control flow with modeled Metal endpoints; no GPU submission."""
import gc
import mmap
import sys
from types import SimpleNamespace
import weakref

import pytest
import triton.backends
from triton_msl.backend import _host_dispatch as lifetime, driver
from triton_msl.errors import PostSubmitError


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    monkeypatch.setattr(lifetime, '_retained', {})
    monkeypatch.setitem(sys.modules, 'Metal', SimpleNamespace(
        MTLSizeMake=lambda *x: x, MTLCommandBufferStatusCompleted=4, MTLCommandBufferStatusError=5,
    ))


class Owner:
    pass


@pytest.mark.parametrize('failure', ['create', 'encode', 'commit', 'wait', 'status', 'error_message', 'interrupt', 'success'])
def test_runtime_failure_retains_owners_until_matching_completion(failure):
    error = KeyboardInterrupt('interrupted wait') if failure == 'interrupt' else RuntimeError(failure)
    events, references = [], {}
    class Encoder:
        def setComputePipelineState_(self, _):
            if failure == 'encode':
                raise error
        def setBuffer_offset_atIndex_(self, *args):
            pass
        def dispatchThreadgroups_threadsPerThreadgroup_(self, *args):
            pass
        def endEncoding(self):
            pass
    class Command:
        final = False
        readable = True
        def computeCommandEncoder(self):
            return Encoder()
        def commit(self):
            events.append('commit')
            assert references['owner']() is not None
            if failure == 'commit':
                raise error
        def waitUntilCompleted(self):
            events.append('wait')
            if failure in ('wait', 'interrupt'):
                raise error
            self.final = True
        def status(self):
            if not self.readable:
                raise RuntimeError('status unreadable')
            if failure == 'status' and 'reap' not in events:
                raise error
            return (5 if failure == 'error_message' else 4) if self.final else 3
        def error(self):
            raise error
    command = Command()
    def create():
        if failure == 'create':
            raise error
        return command
    utils = SimpleNamespace(command_queue=SimpleNamespace(commandBuffer=create))
    def invoke():
        owner, buffer = Owner(), Owner()
        owner.mapping = mmap.mmap(-1, 16384)
        references.update(owner=weakref.ref(owner), mapping=weakref.ref(owner.mapping), buffer=weakref.ref(buffer))
        with lifetime.retain([(buffer, 0)], owners=(owner,)) as bindings:
            driver.MetalUtils.launch(utils, object(), (1, 1, 1), (1, 1, 1), bindings)
    if failure == 'success':
        invoke()
        gc.collect()
        assert not lifetime._retained and all(ref() is None for ref in references.values())
        return
    try:
        invoke()
    except BaseException as caught:
        assert caught is error
        error.__traceback__ = None
    else:
        pytest.fail('expected runtime failure')
    gc.collect()
    assert all(ref() is not None for ref in references.values()), 'owners released before completion proof'
    assert len(lifetime._retained) == 1
    if failure in ('commit', 'wait', 'interrupt'):
        assert lifetime.reap_completed() == 1
        with pytest.raises(PostSubmitError, match='unconfirmed completion'):
            lifetime.check_pending()
        assert events.count('commit') == 1
        command.final = True
        command.readable = False
        assert lifetime.reap_completed() == 1
        command.readable = True
    if failure == 'status':
        assert lifetime.reap_completed() == 1
    events.append('reap')
    error.__traceback__ = None  # Status-query failures must not supply hidden ownership.
    assert lifetime.reap_completed() == 0
    gc.collect()
    assert not lifetime._retained and all(ref() is None for ref in references.values())


@pytest.mark.parametrize('state', [0, 1, 2, 3, 4, 5, 99])
def test_only_terminal_states_retire_a_failed_submission(state):
    owner = Owner()
    ref = weakref.ref(owner)
    try:
        with lifetime.retain([], owners=(owner,)) as buffers:
            buffers.lease.stage = 'submission_attempted'
            buffers.lease.command = SimpleNamespace(status=lambda: state)
            raise RuntimeError('uncertain')
    except RuntimeError:
        pass
    del owner, buffers
    assert lifetime.reap_completed() == (0 if state in (4, 5) else 1)
    gc.collect()
    assert (ref() is None) == (state in (4, 5))


def test_unbound_failure_blocks_more_submissions_without_dropping_owners():
    owner = Owner()
    ref = weakref.ref(owner)
    try:
        with lifetime.retain([], owners=(owner,)):
            raise RuntimeError('custom runtime failed without a command witness')
    except RuntimeError:
        pass
    del owner
    with pytest.raises(PostSubmitError):
        lifetime.check_pending()
    with pytest.raises(PostSubmitError):
        with lifetime.retain([], owners=()):
            pytest.fail('submitted despite quarantine')
    gc.collect()
    assert ref() is not None and len(lifetime._retained) == 1


def test_normal_nested_records_do_not_overwrite_other_owners():
    with lifetime.retain([], owners=(Owner(),)):
        with lifetime.retain([], owners=(Owner(),)):
            assert len(lifetime._retained) == 2
        assert len(lifetime._retained) == 1
    assert not lifetime._retained

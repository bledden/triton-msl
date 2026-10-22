"""Real host marshalling with CPU buffers and modeled dispatch; never submits GPU work."""
from types import SimpleNamespace

import pytest
import torch
import triton.backends

from triton_msl.backend import _cache_contract, _host_extent, _host_dispatch, _launch_contract, driver
from triton_msl.errors import MetalNonRecoverableError, MetalDeviceAssertionError


@pytest.mark.parametrize('mode', [
    'success', 'refusal', 'wrap_failure', 'second_acquire', 'copy_in',
    'scalar_view', 'assertion', 'copy_back', 'exit_hook', 'dispatch_uncertain',
])
def test_host_leases_release_on_safe_boundaries(monkeypatch, mode):
    # Isolate the intentionally uncertain modeled dispatch from later tests.
    monkeypatch.setattr(_host_dispatch, "_retained", {})
    events, acquired, released = [], [], []
    failure = RuntimeError('controlled ' + mode)

    class Buffer:
        def __init__(self, size, scalar=False):
            self.data = bytearray(size)
            self.scalar = scalar
        def contents(self):
            return self
        def as_buffer(self, size):
            if mode == 'scalar_view' and self.scalar:
                raise failure
            if mode == 'copy_in' and self is acquired[0][0]:
                raise failure
            if mode == 'copy_back' and 'completed' in events and not self.scalar:
                raise failure
            return memoryview(self.data)[:size]

    class Pool:
        def acquire(self, size):
            if mode == 'second_acquire' and acquired:
                raise failure
            row = (Buffer(size), object(), size)
            acquired.append(row)
            return row
        def acquire_scalar(self, size):
            buf = Buffer(size, scalar=True)
            acquired.append(('scalar', buf, size))
            return buf
        def release(self, *entry):
            released.append(entry)
        def release_scalar(self, buf, size):
            released.append(('scalar', buf, size))

    selections = []
    def select(storage, ptr, length, *args, **kwargs):
        selections.append(ptr)
        if len(selections) == 2:
            if mode == 'refusal':
                return 'refuse', 0, 'controlled owner mismatch'
            if mode == 'wrap_failure':
                return 'borrow', length, 'admitted'
        return 'mirror', length, 'controlled mirror'

    def wrap(*args):
        raise failure
    def launch(*args):
        events.append('dispatch')
        if mode == 'dispatch_uncertain':
            raise failure
        if mode == 'assertion':
            args[3][-1][0].data[:4] = (1).to_bytes(4, 'little')
        events.append('completed')
    def leave(_):
        assert released == acquired
        if mode == 'exit_hook':
            raise failure

    monkeypatch.setenv('TRITON_MSL_COMPILE_SHADER', '0')
    monkeypatch.setattr(_cache_contract, 'validate_execution_contract', lambda _: None)
    monkeypatch.setattr(_launch_contract, 'validate_packed_launch', lambda metadata, _: metadata)
    monkeypatch.setattr(_host_extent, 'select', select)
    pool = Pool()
    utils = SimpleNamespace(buffer_pool=pool, launch=launch, make_buffer_from_ptr=wrap,
                            make_buffer_with_data=lambda data, size: Buffer(size))
    monkeypatch.setattr(driver, '_get_utils', lambda: utils)
    launcher = driver.MetalLauncher.__new__(driver.MetalLauncher)
    launcher.arg_names = ['x', 'y', 'n']
    launcher.signature = dict(x='*fp32', y='*fp32', n='i32')
    launcher.constexpr_indices = set()
    launcher._msl = None
    desc = dict(buffer_index=3, messages=['controlled assertion']) if mode == 'assertion' else None
    metadata = (4, 1, 0, 32, [1], False, None, None, None, None, None, desc)
    x, y = torch.ones(17), torch.zeros(17)
    call = lambda: launcher(1, 1, 1, None, None, metadata, None, None, leave, x, y, 17)
    if mode == 'success':
        call()
    else:
        error = MetalNonRecoverableError if mode == 'refusal' else MetalDeviceAssertionError if mode == 'assertion' else RuntimeError
        with pytest.raises(error) as raised:
            call()
        if mode not in ('refusal', 'assertion'):
            assert raised.value is failure
    assert acquired
    assert released == ([] if mode == 'dispatch_uncertain' else acquired)
    assert len({id(row[1] if row[0] == 'scalar' else row[0]) for row in released}) == len(released)
    if mode in ('refusal', 'wrap_failure', 'second_acquire', 'copy_in', 'scalar_view'):
        assert 'dispatch' not in events
    else:
        assert events.count('dispatch') == 1


@pytest.mark.parametrize('active', [False, True])
def test_cleanup_attempts_all_entries_and_preserves_primary_failure(active):
    visited = []
    cleanup = RuntimeError('cleanup failure')
    primary = ValueError('primary failure')
    def release(*entry):
        visited.append(entry)
        if len(visited) == 1:
            raise cleanup
    pool = SimpleNamespace(release=release)
    entries = [(object(), object(), 4), (object(), object(), 8)]
    if active:
        with pytest.raises(ValueError) as raised:
            try:
                raise primary
            finally:
                driver._release_host_pool_buffers(pool, entries)
        assert raised.value is primary
        assert primary.__notes__ == ['Host buffer cleanup also failed: cleanup failure']
    else:
        with pytest.raises(RuntimeError) as raised:
            driver._release_host_pool_buffers(pool, entries)
        assert raised.value is cleanup
    assert visited == entries


def test_cleanup_data_entries_never_dispatch_equality():
    comparisons, released = [], []

    class Buffer:
        def __eq__(self, other):
            comparisons.append((id(self), other))
            return False

    buffers = [Buffer() for _ in range(3)]
    scalar = object()
    entries = [(buffers[0], object(), 4), ("scalar", scalar, 4),
               (buffers[1], object(), 8), (buffers[2], object(), 16)]
    pool = SimpleNamespace(
        release=lambda *entry: released.append(("data", tuple(id(v) for v in entry))),
        release_scalar=lambda *entry: released.append(("scalar", tuple(id(v) for v in entry))),
    )
    driver._release_host_pool_buffers(pool, entries)
    assert comparisons == []
    assert released == [
        ("data", tuple(id(v) for v in entries[0])),
        ("scalar", (id(scalar), id(entries[1][2]))),
        ("data", tuple(id(v) for v in entries[2])),
        ("data", tuple(id(v) for v in entries[3])),
    ]

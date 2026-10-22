"""Real mixed mirror/borrow preparation failures return earlier pool leases."""
import ctypes
import mmap

import numpy as np
import pytest
import torch

from tests.test_host_extent_gpu import add, data, launch_checked
from triton_msl.backend import _cache_contract, driver
from triton_msl.errors import MetalNonRecoverableError


@pytest.mark.parametrize('mode', ['refusal', 'wrap_failure', 'copy_in'])
def test_mirror_before_borrow_failure_returns_lease(monkeypatch, mode):
    assert torch.backends.mps.is_available() and _cache_contract._HOST_EXTENT_READY
    monkeypatch.setenv('TRITON_MSL_COMPILE_SHADER', '0')
    monkeypatch.setenv('TRITON_MSL_HOST_EXTENT', '1')
    x, y, backing, out, n = data()
    mapping = mmap.mmap(-1, x.numel() * 4)
    array = np.ndarray((x.numel(),), dtype=np.float32, buffer=mapping)
    array.fill(2)
    x = torch.from_numpy(array)
    launch_checked(x, y, backing, out, n)
    utils = driver._get_utils()
    pool = utils.buffer_pool
    acquire, release = pool.acquire, pool.release
    acquired, released = [], []
    def take(size):
        row = acquire(size)
        acquired.append(row)
        return row
    def give(*row):
        released.append(row)
        return release(*row)
    monkeypatch.setattr(pool, 'acquire', take)
    monkeypatch.setattr(pool, 'release', give)
    failure = MemoryError('controlled preparation failure')
    def failed(*args):
        raise failure
    def forbidden(*args):
        raise AssertionError('dispatch after failed preparation')
    monkeypatch.setattr(utils, 'launch', forbidden)
    if mode == 'refusal':
        query = _cache_contract._validation_native.host_extent
        storage = y.untyped_storage()
        ptr = storage.data_ptr()
        def mutate(pointer, length):
            result = query(pointer, length)
            if pointer == ptr:
                storage.resize_(length + 4)
            return result
        monkeypatch.setattr(_cache_contract._validation_native, 'host_extent', mutate)
    elif mode == 'wrap_failure':
        monkeypatch.setattr(utils, 'make_buffer_from_ptr', failed)
    else:
        monkeypatch.setattr(ctypes, 'memmove', failed)
    with pytest.raises(MetalNonRecoverableError if mode == 'refusal' else MemoryError) as caught:
        add[((n + 255) // 256,)](x, y, out, n, BLOCK=256)
    if mode != 'refusal':
        assert caught.value is failure
    assert len(acquired) == len(released) == 1
    assert released[0][0] is acquired[0][0] and released[0][1] is acquired[0][1]
    assert any(buf is acquired[0][0] for buf, _ in pool._free[acquired[0][2]])

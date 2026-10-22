"""Actual host-route extent controls; the evidence runner holds the shared GPU lock."""
import ctypes
import gc
import mmap
import sys

import numpy as np
import pytest
import torch
import triton
import triton.language as tl

from triton_msl.backend import _cache_contract as contract, _host_extent as extent, driver
from triton_msl.errors import MetalNonRecoverableError


@triton.jit
def add(X, Y, O, N, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(O + i, tl.load(X + i, i < N, 0) + tl.load(Y + i, i < N, 0), i < N)


@triton.jit
def two_outputs(A, B, N, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(B + i, 2.0, i < N)
    tl.store(A + i, 3.0, i < N)


@pytest.fixture(autouse=True)
def host(monkeypatch):
    assert torch.backends.mps.is_available(), "runner must provide real Metal"
    assert contract._HOST_EXTENT_READY, "runner must load the extent-equipped helper"
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "0")
    monkeypatch.setenv("TRITON_MSL_HOST_EXTENT", "1")


def data(n=1048579):
    alloc = triton.cdiv(n, 256) * 256
    x = torch.full((alloc,), 2.0)
    y = torch.full((alloc,), 5.0)
    backing = torch.full((alloc + 32,), -12345.5)
    return x, y, backing, backing[16:-16], n


def launch_checked(x, y, backing, out, n):
    before_x = ctypes.string_at(x.data_ptr(), x.numel() * 4)
    before_y = ctypes.string_at(y.data_ptr(), y.numel() * 4)
    expected = backing.numpy().copy()
    expected[16:16+n] = np.frombuffer(before_x, dtype=np.float32)[:n] + np.frombuffer(before_y, dtype=np.float32)[:n]
    compiled = add[(triton.cdiv(n, 256),)](x, y, out, n, BLOCK=256)
    assert backing.numpy().tobytes() == expected.tobytes()
    assert ctypes.string_at(x.data_ptr(), x.numel() * 4) == before_x
    assert ctypes.string_at(y.data_ptr(), y.numel() * 4) == before_y
    return compiled


def count_call(call, monkeypatch):
    original = ctypes.memmove
    copies = []
    def forward(dst, src, n):
        copies.append(n)
        return original(dst, src, n)
    with monkeypatch.context() as m:
        m.setattr(ctypes, "memmove", forward)
        call()
    return copies


def test_large_actual_borrow_and_fresh_cpu_write(monkeypatch):
    x, y, backing, out, n = data()
    launch_checked(x, y, backing, out, n)
    x[0] = 17
    copies = count_call(lambda: launch_checked(x, y, backing, out, n), monkeypatch)
    assert copies == []
    assert len(extent.last_launch()) == 3
    assert all(row['route'] == 'borrow' for row in extent.last_launch())


def test_small_has_no_extent_query(monkeypatch):
    values = data(257)
    launch_checked(*values)
    def forbidden(*args):
        raise AssertionError("small geometry queried allocator")
    monkeypatch.setattr(contract._validation_native, 'host_extent', forbidden)
    assert count_call(lambda: launch_checked(*values), monkeypatch) == [2048, 2048, 2176, 2176]
    assert all(row['reason'] == 'geometry' for row in extent.last_launch())


@pytest.mark.parametrize('kind', ['mmap', 'interior'])
def test_mixed_foreign_or_interior_keeps_only_its_mirror(monkeypatch, kind):
    x, y, backing, out, n = data()
    if kind == 'mmap':
        owner = mmap.mmap(-1, x.numel() * 4)
        array = np.ndarray((x.numel(),), dtype=np.float32, buffer=owner)
    else:
        owner = np.empty(x.numel() + 4096, dtype=np.float32)
        array = owner[4096:]
    array.fill(2)
    x = torch.from_numpy(array)
    launch_checked(x, y, backing, out, n)
    assert count_call(lambda: launch_checked(x, y, backing, out, n), monkeypatch) == [x.numel() * 4]
    assert extent.last_launch()[0]['reason'] == 'no_malloc_extent'


def test_delegating_custom_handler_gpu_tail_and_lifetime(monkeypatch):
    probe = sys.modules.get('_public_extent_probe')
    assert probe is not None, "runner must load the retained custom allocator before identity"
    before = probe.stats()
    old = probe.install()
    try:
        array = np.empty(1048864, dtype=np.float32)
    finally:
        probe.restore(old)
    assert np._core.multiarray.get_handler_name(array) == 'extent_custom_default_backing'
    array.fill(-12345.5)
    pointer, size = array.ctypes.data, array.nbytes
    tail = ctypes.string_at(pointer + size, 128)
    backing = torch.from_numpy(array)
    out = backing[16:-16]
    del array
    gc.collect()
    assert probe.stats()['frees'] == before['frees']
    x, y, _, _, n = data()
    launch_checked(x, y, backing, out, n)
    assert count_call(lambda backing=backing, out=out: launch_checked(x, y, backing, out, n), monkeypatch) == []
    assert ctypes.string_at(pointer + size, 128) == tail == bytes([0xa7]) * 128
    assert extent.last_launch()[2]['route'] == 'borrow'
    del backing, out
    gc.collect()
    after = probe.stats()
    assert after['allocations'] == before['allocations'] + 1
    assert after['frees'] == before['frees'] + 1
    assert after['bad_tails'] == before['bad_tails']


def test_borrow_preserves_exact_alias_and_two_output_order():
    x, y, backing, out, n = data()
    out[:n] = 2
    add[(triton.cdiv(n, 256),)](out, y, out, n, BLOCK=256)
    assert torch.equal(out[:n], torch.full((n,), 7.0))
    assert any(row['members'] == (0, 2) for row in extent.last_launch())
    two_outputs[(triton.cdiv(n, 256),)](out, out.view_as(out), n, BLOCK=256)
    assert torch.equal(out[:n], torch.full((n,), 3.0))
    assert backing[:16].eq(-12345.5).all() and backing[16+n:].eq(-12345.5).all()
    assert extent.last_launch()[0]['route'] == 'borrow'


def test_fresh_owner_replacement_and_resize():
    x, y, backing, out, n = data()
    launch_checked(x, y, backing, out, n)
    x = torch.full_like(x, 9)
    launch_checked(x, y, backing, out, n)
    storage = x.untyped_storage()
    storage.resize_(storage.nbytes() + 4)
    launch_checked(x, y, backing, out, n)
    assert extent.last_launch()[0]['storage_nbytes'] == storage.nbytes()


@pytest.mark.parametrize('mutation', ['resize', 'replace'])
def test_changed_owner_refuses_before_copy_or_dispatch(monkeypatch, mutation):
    x, y, backing, out, n = data()
    launch_checked(x, y, backing, out, n)
    storage = x.untyped_storage()
    query = contract._validation_native.host_extent
    def change(ptr, length):
        result = query(ptr, length)
        if ptr == storage.data_ptr():
            if mutation == 'resize':
                storage.resize_(length + 4)
            else:
                x.set_(torch.full_like(x, 19))
        return result
    def forbidden(*args, **kwargs):
        raise AssertionError('dispatch after owner change')
    monkeypatch.setattr(contract._validation_native, 'host_extent', change)
    monkeypatch.setattr(driver._get_utils(), 'launch', forbidden)
    monkeypatch.setattr(ctypes, 'memmove', forbidden)
    with pytest.raises(MetalNonRecoverableError, match='storage_changed_during_query'):
        add[(triton.cdiv(n, 256),)](x, y, out, n, BLOCK=256)
    assert extent.last_launch()[0]['route'] == 'refuse'


def test_borrow_allocation_failure_propagates_before_dispatch(monkeypatch):
    values = data()
    launch_checked(*values)
    failure = MemoryError('extent wrapper allocation failed')
    def failed(*args):
        raise failure
    def forbidden(*args, **kwargs):
        raise AssertionError('dispatch after allocation failure')
    monkeypatch.setattr(driver._get_utils(), 'make_buffer_from_ptr', failed)
    monkeypatch.setattr(driver._get_utils(), 'launch', forbidden)
    with pytest.raises(MemoryError) as raised:
        launch_checked(*values)
    assert raised.value is failure


def test_exact_page_preserves_existing_path(monkeypatch):
    n = 1048576
    x, y, out = (torch.full((n,), v) for v in (2.0, 5.0, -3.0))
    add[(triton.cdiv(n, 256),)](x, y, out, n, BLOCK=256)
    def forbidden(*args):
        raise AssertionError('exact-page path queried extent')
    monkeypatch.setattr(contract._validation_native, 'host_extent', forbidden)
    assert count_call(lambda: add[(triton.cdiv(n, 256),)](x, y, out, n, BLOCK=256), monkeypatch) == []
    assert torch.equal(out, torch.full_like(out, 7))
    assert all(row['route'] == 'existing' for row in extent.last_launch())


def test_mps_keeps_native_route(monkeypatch):
    monkeypatch.setenv('TRITON_MSL_COMPILE_SHADER', '1')
    x, y, out = (torch.full((257,), v, device='mps') for v in (2.0, 5.0, -3.0))
    def forbidden(*args, **kwargs):
        raise AssertionError('MPS entered host launch')
    monkeypatch.setattr(driver._get_utils(), 'launch', forbidden)
    add[(2,)](x, y, out, 257, BLOCK=256)
    torch.mps.synchronize()
    assert torch.equal(out.cpu(), torch.full((257,), 7.0))
    assert extent.last_launch() is None


@triton.jit
def guarded_store(O, valid):
    tl.device_assert(valid != 0, "borrow guard")
    tl.store(O + tl.arange(0, 8), 23.0)


def test_borrowed_output_assertion_failure_then_success():
    from triton_msl.errors import MetalDeviceAssertionError
    _, _, backing, out, _ = data()
    before = backing.numpy().tobytes()
    with pytest.raises(MetalDeviceAssertionError, match="borrow guard"):
        guarded_store[(1,)](out, 0, debug=True)
    assert backing.numpy().tobytes() == before
    assert extent.last_launch()[0]['route'] == 'borrow'
    guarded_store[(1,)](out, 2, debug=True)
    assert out[:8].eq(23).all()
    assert backing[:16].eq(-12345.5).all() and backing[24:].eq(-12345.5).all()
    assert extent.last_launch()[0]['route'] == 'borrow'

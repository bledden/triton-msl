"""Page-aligned Metal buffer pool for reducing per-kernel allocation overhead.

The pool maintains page-aligned memory regions that can be wrapped with
newBufferWithBytesNoCopy for zero-copy Metal buffer creation. This avoids
the per-call overhead of Metal buffer allocation and data copying.

Size classes: buffers are rounded up to the nearest power of 2,
minimum 16KB (ARM64 page size). Each size class maintains a free list.

Usage:
    pool = MetalBufferPool(metal_device)
    buf, aligned_mem = pool.acquire(nbytes)
    # ... use buf as Metal buffer ...
    pool.release(buf, aligned_mem, size_class)
"""

import ctypes
import mmap
from collections import defaultdict, OrderedDict


# ARM64 page size
PAGE_SIZE = 16384


def _round_up_power_of_2(n):
    """Round up to the nearest power of 2, minimum PAGE_SIZE."""
    n = max(n, PAGE_SIZE)
    # Bit trick: round up to next power of 2
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    n += 1
    return n


class MetalBufferPool:
    """Pool of page-aligned Metal buffers for zero-copy kernel dispatch."""

    def __init__(self, device):
        self._device = device
        # Free lists keyed by size class
        self._free: dict[int, list[tuple]] = defaultdict(list)
        # Max buffers per size class to prevent unbounded growth
        self._max_per_class = 8
        # Small scalar buffer pool (4 bytes, 8 bytes)
        self._scalar_free: dict[int, list] = defaultdict(list)
        self._max_scalars = 32
        # Buffer cache: maps (data_ptr, nbytes) → (metal_buf, aligned_mem, size_class)
        # Avoids re-copying read-only tensors that haven't changed between kernel launches
        self._cache: OrderedDict = OrderedDict()
        self._cache_max = 32

    def acquire(self, nbytes):
        """Acquire a page-aligned Metal buffer of at least nbytes.

        Returns: (metal_buffer, aligned_mmap, size_class)
            metal_buffer: MTLBuffer wrapping the aligned memory (zero-copy)
            aligned_mmap: mmap object (must be kept alive while buffer is in use)
            size_class: the actual size allocated (for release)
        """
        size_class = _round_up_power_of_2(nbytes)

        # Check free list first
        free_list = self._free[size_class]
        if free_list:
            metal_buf, aligned_mem = free_list.pop()
            return metal_buf, aligned_mem, size_class

        # Allocate new page-aligned memory via mmap
        aligned_mem = mmap.mmap(-1, size_class)

        # Get the actual pointer address
        buf_type = ctypes.c_char * size_class
        buf_from_mem = buf_type.from_buffer(aligned_mem)
        ptr = ctypes.addressof(buf_from_mem)

        # Wrap with newBufferWithBytesNoCopy (zero-copy)
        import Metal
        metal_buf = self._device.newBufferWithBytesNoCopy_length_options_deallocator_(
            buf_from_mem,
            size_class,
            Metal.MTLResourceStorageModeShared,
            None,
        )
        if metal_buf is None:
            # Fallback: allocate via Metal API (loses zero-copy benefit)
            aligned_mem.close()
            metal_buf = self._device.newBufferWithLength_options_(
                size_class, Metal.MTLResourceStorageModeShared
            )
            return metal_buf, None, size_class

        return metal_buf, aligned_mem, size_class

    def release(self, metal_buf, aligned_mem, size_class):
        """Return a buffer to the pool for reuse."""
        if aligned_mem is None:
            # Metal-allocated buffer, can't reuse via pool
            return

        free_list = self._free[size_class]
        if len(free_list) < self._max_per_class:
            free_list.append((metal_buf, aligned_mem))
        else:
            # Pool is full for this size class — let it be GC'd
            aligned_mem.close()

    def acquire_scalar(self, nbytes):
        """Acquire a small Metal buffer for scalar arguments (4 or 8 bytes).

        Returns: metal_buffer
        """
        free_list = self._scalar_free[nbytes]
        if free_list:
            return free_list.pop()

        import Metal
        buf = self._device.newBufferWithLength_options_(
            max(nbytes, 4), Metal.MTLResourceStorageModeShared
        )
        return buf

    def release_scalar(self, buf, nbytes):
        """Return a scalar buffer to the pool."""
        free_list = self._scalar_free[nbytes]
        if len(free_list) < self._max_scalars:
            free_list.append(buf)

    def acquire_cached(self, data_ptr, nbytes):
        """Try to get a cached Metal buffer for a tensor.

        Returns (metal_buf, aligned_mem, size_class) if cached, None otherwise.
        Moves the entry to most-recent position (LRU).
        """
        key = (data_ptr, nbytes)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def cache_buffer(self, data_ptr, nbytes, metal_buf, aligned_mem, size_class):
        """Cache a Metal buffer for a tensor's data_ptr.

        Evicts least-recently-used entry if cache is full.
        """
        key = (data_ptr, nbytes)
        if key in self._cache:
            self._cache.move_to_end(key)
            return
        if len(self._cache) >= self._cache_max:
            # Evict LRU entry — release buffer back to pool
            _, (old_buf, old_mem, old_sc) = self._cache.popitem(last=False)
            self.release(old_buf, old_mem, old_sc)
        self._cache[key] = (metal_buf, aligned_mem, size_class)

    def invalidate_cache(self, data_ptr, nbytes):
        """Invalidate a cached buffer (tensor was written to by kernel)."""
        key = (data_ptr, nbytes)
        if key in self._cache:
            del self._cache[key]

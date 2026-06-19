"""Zero-copy MPS execution via torch.mps.compile_shader.

PyTorch (newer versions) can compile a Metal compute library from MSL source
and dispatch its kernels against MPS tensors zero-copy (PyTorch owns the
buffers + the MPS stream). This runtime wraps that: compile (cached on the MSL
string) + dispatch. It has NO triton-msl driver knowledge — the driver
selects when to use it.
"""
from __future__ import annotations


class CompileShaderRuntime:
    """Compile + cache + dispatch MSL via torch.mps.compile_shader."""

    def __init__(self):
        self._lib_cache = {}               # msl_source -> compiled library
        self._unsupported: set[str] = set()  # msl_source strings that failed; skip fast-path

    def available(self) -> bool:
        try:
            import torch
            return (hasattr(torch, "mps")
                    and torch.backends.mps.is_available()
                    and hasattr(torch.mps, "compile_shader"))
        except Exception:
            return False

    def is_unsupported(self, msl: str) -> bool:
        return msl in self._unsupported

    def mark_unsupported(self, msl: str) -> None:
        self._unsupported.add(msl)

    def get_library(self, msl: str):
        """Compile MSL (cached on the source string). Raises on compile error."""
        lib = self._lib_cache.get(msl)
        if lib is None:
            import torch
            lib = torch.mps.compile_shader(msl)
            self._lib_cache[msl] = lib
        return lib

    def dispatch(self, lib, kernel_name: str, args, *, threads, group_size) -> None:
        """Dispatch lib.<kernel_name>(*args, threads=..., group_size=...).

        ``args`` is the ordered argument list (MPS tensors + Python scalars)
        matching the kernel's [[buffer(i)]] order. ``threads`` and ``group_size``
        are ints (1-D) or tuples (2-D/3-D). PyTorch binds the MPS tensors
        zero-copy and enqueues on the MPS stream.
        """
        getattr(lib, kernel_name)(*args, threads=threads, group_size=group_size)

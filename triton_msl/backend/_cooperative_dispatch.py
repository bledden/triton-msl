"""Default-off zero-copy candidate for the proved cooperative attention family.

The compiler option creates the sealed descriptor; ordinary compilations never
call this path. It is for per-route numerical qualification, not an assertion
that the direct-Metal experiment qualifies PyTorch's compiler options.
"""

from types import SimpleNamespace

from ._attention_variant import PipelineFacts, select_variant, validate_variant_geometry


def layout_eligible(spans, *, nq, nk, groups):
    """Integer-only reach/alias check after extracting exact tensor properties.

    spans are (opaque MTLBuffer identity, byte offset, visible bytes) in role
    order. PyTorch's MPS storage data pointer names the buffer object, not its
    contents. Distinct handles must never be compared as memory intervals.
    Unknown or overlapping ranges keep scalar, except single-program Bias/Out.
    No claim about baseline cross-program alias safety is made here.
    """
    if any(type(x) is not int or not 0 < x <= 0x7FFFFFFF for x in (nq, nk, groups)):
        return False
    # Preserve the scalar source's uint index domain; avoid a new address proof.
    if max(nq * 64, nk * 64, nq * nk, groups * 32) > 0x7FFFFFFF:
        return False
    required = (nq * 128, nk * 128, nk * 128, nq * nk * 4, nq * 256)
    if len(spans) != 5:
        return False
    for (buffer, offset, size), need in zip(spans, required):
        if (
            type(buffer) is not int
            or type(offset) is not int
            or type(size) is not int
            or buffer <= 0
            or offset < 0
            or size < need
        ):
            return False
    out_buffer, out_offset, _ = spans[4]
    out_end = out_offset + required[4]
    for i, ((buffer, offset, _), size) in enumerate(zip(spans[:4], required[:4])):
        overlap = buffer == out_buffer and out_offset < offset + size and offset < out_end
        if overlap and not (i == 3 and groups == 1):
            return False
    return True


def _tensor_layout(descriptor, args, grid):
    import torch

    if type(args) not in (list, tuple) or len(args) != descriptor["arg_count"]:
        return False
    if len(grid) != 3 or any(type(x) is not int for x in grid) or grid[1:] != (1, 1):
        return False
    roles = [args[i] for i in descriptor["roles"]]
    tensors, nq, nk = roles[:5], roles[5], roles[6]
    spans = []
    for tensor, dtype in zip(tensors, (torch.float16, torch.float16, torch.float16, torch.float32, torch.float32)):
        if (
            type(tensor) is not torch.Tensor
            or tensor.device.type != "mps"
            or tensor.dtype != dtype
            or tensor.layout != torch.strided
            or not tensor.is_contiguous()
            or tensor.is_conj()
            or tensor.is_neg()
        ):
            return False
        storage = tensor.untyped_storage()
        buffer = storage.data_ptr()
        address = tensor.data_ptr()
        size = tensor.numel() * tensor.element_size()
        offset = tensor.storage_offset() * tensor.element_size()
        if offset < 0 or address != buffer + offset or offset + size > storage.nbytes():
            return False
        spans.append((buffer, offset, size))
    return layout_eligible(spans, nq=nq, nk=nk, groups=grid[0])


def dispatch_cooperative_attention(
    rt,
    descriptor,
    args,
    grid,
    *,
    device_limits,
    submission_state,
    launch_exit_hook=None,
    launch_metadata=None,
):
    """Return False only before invocation; the caller retains scalar execution."""
    if not _tensor_layout(descriptor, args, grid):
        return False
    source, entry = descriptor["source"], descriptor["entry"]
    if rt.is_unsupported(source):
        return False
    try:
        library = rt.get_library(source)
        kernel = getattr(library, entry)
        facts = PipelineFacts(
            kernel.thread_execution_width,
            kernel.max_threads_per_threadgroup,
            kernel.static_thread_group_memory_length,
        )
        # This is the explicit experimental route, not a release qualification.
        selection = select_variant(
            facts,
            device_max_threads=device_limits[0],
            device_max_shared=device_limits[1],
            route_enabled=True,
        )
        if not selection.cooperative:
            return False
        validate_variant_geometry(selection, threads=descriptor["threads"], shared=descriptor["shared"])
        # Retain the very callable whose facts were read; dispatch's getattr
        # cannot resolve a replacement from the original library afterward.
        resolved = SimpleNamespace(**{entry: kernel})
    except Exception:
        # A compile/property failure is before submission. Do not poison the
        # scalar stash or mark a possibly already submitted operation retryable.
        rt.mark_unsupported(source)
        return False
    submission_state.begin()
    rt.dispatch(resolved, entry, args, threads=grid[0] * selection.threads, group_size=selection.threads)
    if launch_exit_hook:
        launch_exit_hook(launch_metadata)
    return True

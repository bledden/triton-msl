"""Actual integrated bytes for the two capability-parity witnesses; no compiler monkeypatch."""

import hashlib

import numpy as np
import pytest
import triton  # noqa: F401

from triton_msl.backend.compiler import MetalBackend, MetalOptions
from triton_msl.backend.driver import _get_utils
from test_cpp_cfg_integration import CHAINED, CORPUS


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("name", ["TWO_PTRS_SAME_MASK", "CHAINED_ADDPTR_F32"])
def test_fallback_artifacts_execute(monkeypatch, tmp_path, record_property, enabled, name):
    import torch

    if not torch.backends.mps.is_available():
        pytest.skip("Metal hardware unavailable")
    monkeypatch.setenv("TRITON_MSL_CPP_SCALAR_CFG", "1" if enabled else "0")
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "cache"))
    source = CHAINED if name == "CHAINED_ADDPTR_F32" else CORPUS[name]
    metadata = {}
    options = MetalOptions(num_warps=4)
    typed = MetalBackend.make_llir(source, metadata, options)
    assert metadata["cpp_scalar_cfg_route"] == ("fallback" if enabled else "disabled")
    assert bool(metadata["cpp_scalar_cfg_reason"]) == enabled
    binary = MetalBackend.make_metallib_from_llir(typed, metadata, options)
    digest = hashlib.sha256(binary).hexdigest()
    (tmp_path / "kernel.ll").write_text(typed)
    (tmp_path / "kernel.metallib").write_bytes(binary)
    utils = _get_utils()
    _, pipeline, *_ = utils.load_binary(metadata["name"], binary, 0)
    x = torch.arange(160, dtype=torch.float32) * 3 + 1
    out = torch.full((160,), -999.0, dtype=torch.float32)
    n = torch.zeros(1, dtype=torch.int32)
    flag = torch.ones(1, dtype=torch.int32)
    tensors = [x, out, n, flag]
    buffers = [(utils.make_buffer_from_ptr(t.data_ptr(), t.numel() * t.element_size()), 0) for t in tensors]
    # The splat-output fixture writes one cell. One lane avoids a cross-lane
    # race; the CPU checker separately enumerates the fixture's lane semantics.
    threads = 128 if name == "CHAINED_ADDPTR_F32" else 1
    for bound in [-1, 0, 1, 63, 127, 128]:
        n.fill_(bound)
        out.fill_(-999.0)
        utils.launch(pipeline, (1, 1, 1), (threads, 1, 1), buffers, sync=True)
        expected = np.full(160, -999.0, dtype=np.float32)
        if name == "CHAINED_ADDPTR_F32":
            count = max(0, min(bound, 128))
            expected[:count] = x.numpy()[1 : 1 + count]
        else:
            expected[0] = float(x[0] + x[1]) if bound > 0 else 14.0
        assert out.numpy().tobytes() == expected.tobytes(), (name, bound, enabled)
        assert np.array_equal(x.numpy(), np.arange(160, dtype=np.float32) * 3 + 1)
    record_property("cpp_scalar_cfg_route", metadata["cpp_scalar_cfg_route"])
    record_property("cpp_scalar_cfg_reason", metadata["cpp_scalar_cfg_reason"])
    record_property("metallib_sha256", digest)
    record_property("typed_ir_sha256", hashlib.sha256(typed.encode()).hexdigest())
    record_property("artifact_directory", str(tmp_path))
    record_property("checked_bound_cases", 6)

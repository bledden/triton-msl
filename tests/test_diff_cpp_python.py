"""Route-verified differential on a bounded experimental C++ elementwise slice.

Each path runs in its own subprocess (fresh cache dir, fresh process state)
and saves the kernel output buffer plus its binary/dispatch receipt. Neither
MSL fallback nor compile_shader dispatch earns C++ correctness credit.
Skips entirely when the C++ pass library isn't built.
The historical fp32-to-fp16 case remains as an explicitly labelled MSL-fallback
control, not evidence of C++ half-precision coverage.
"""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import triton  # noqa: F401  (backend discovery before triton_msl imports)
from triton_msl.backend.compiler import MetalBackend

pytestmark = pytest.mark.skipif(not MetalBackend._has_cpp_passes(), reason="cpp not built")

KERNEL = """import hashlib, importlib.util, json, os, sys
from pathlib import Path
import numpy as np, torch, triton, triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
import triton_msl
from triton_msl.backend import compiler, driver

output, route, image, dtype, block, groups = sys.argv[1:]
block, groups = int(block), int(groups)
half_output = dtype == "fp32_to_fp16"
expected_route = "msl" if half_output else route
if not torch.backends.mps.is_available():
    raise RuntimeError("Metal unavailable; no numerical receipt")
if route == "cpp":
    # Bind this arm to the same extension the parent actually loaded, including
    # an isolated build outside the package. Never silently substitute another.
    spec = importlib.util.spec_from_file_location("triton_msl._triton_msl_cpp", image)
    cpp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cpp)
    sys.modules[spec.name] = cpp
    setattr(triton_msl, "_triton_msl_cpp", cpp)
    cpp.register_metal_passes()

@triton.jit
def k(X, O, BLOCK: tl.constexpr, HALF_OUTPUT: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i)
    result = x * 2 + 1
    if HALF_OUTPUT:
        result = result.to(tl.float16)
    tl.store(O + i, result)

src = ASTSource(
    fn=k, signature={"X": "*fp32" if half_output else "*" + dtype, "O": "*fp16" if half_output else "*" + dtype},
    constexprs={"BLOCK": block, "HALF_OUTPUT": half_output},
)
kernel = triton.compile(src, target=GPUTarget("metal", "apple-m4", 32))
assert kernel.metadata.binary_route == expected_route, (route, kernel.metadata)
if route == "cpp":
    if half_output:
        assert kernel.metadata.cpp_fallback_reason
    else:
        assert kernel.metadata.cpp_fallback_reason is None
n = block * groups
tdtype = torch.float32 if dtype in ("fp32", "fp32_to_fp16") else torch.int32
x = ((torch.arange(n, dtype=torch.int32) % 127) - 63).to(tdtype)
if half_output:
    torch.manual_seed(0)
    x = torch.randn(n)
output_type = torch.float16 if half_output else tdtype
backing = torch.full((n + 32,), -999, dtype=output_type)
out = backing[16:16 + n]
utils = driver._get_utils()
launch = utils.launch
dispatches = []
def tracked_launch(pipeline, grid, threadgroup_size, buffers, *args, **kwargs):
    dispatches.append({"groups": list(grid), "group_size": list(threadgroup_size)})
    return launch(pipeline, grid, threadgroup_size, buffers, *args, **kwargs)
utils.launch = tracked_launch
kernel[(groups, 1, 1)](x, out)
assert len(dispatches) == 1, dispatches
expected = np.asarray(x.numpy(), dtype=np.float64) * 2 + 1
np.testing.assert_array_equal(out.numpy(), expected.astype(out.numpy().dtype))
assert torch.equal(backing[:16], torch.full((16,), -999, dtype=output_type))
assert torch.equal(backing[-16:], torch.full((16,), -999, dtype=output_type))
np.save(output, backing.numpy())
binary_hash = hashlib.sha256(kernel.asm["metallib"]).hexdigest()
receipt = {
    "route": kernel.metadata.binary_route, "requested_route": route, "dispatches": dispatches,
    "metallib_sha256": binary_hash, "dtype": dtype, "block": block, "groups": groups,
    "extension_sha256": hashlib.sha256(Path(image).read_bytes()).hexdigest() if route == "cpp" else None,
    "reference": "float64 affine expression rounded to output dtype", "canaries": "intact",
}
Path(output + ".json").write_text(json.dumps(receipt, indent=2))
"""


def _run(directory, force_python, dtype, block, groups):
    import triton_msl._triton_msl_cpp as cpp

    directory.mkdir()
    source = directory / "kernel.py"
    source.write_text(KERNEL)
    out = directory / "output.npy"
    # Isolate both cache layers independently of their policy-key guarantees.
    env = dict(
        os.environ,
        PYTHONPATH=str(Path(__file__).resolve().parents[1]),
        TRITON_MSL_CACHE_DIR=str(directory / "msl-cache"),
        TRITON_CACHE_DIR=str(directory / "triton-cache"),
        TRITON_MSL_COMPILE_SHADER="0",
        TRITON_ALWAYS_COMPILE="1",
    )
    # C++ is opt-in (default-on flip reverted): the C++ side must request it
    # explicitly, else both sides route Python and the differential is vacuous.
    if force_python:
        env["TRITON_MSL_FORCE_PYTHON"] = "1"
        env.pop("TRITON_MSL_USE_CPP", None)
    else:
        env.pop("TRITON_MSL_FORCE_PYTHON", None)
        env["TRITON_MSL_USE_CPP"] = "1"
    route = "msl" if force_python else "cpp"
    result = subprocess.run(
        [sys.executable, str(source), str(out), route, cpp.__file__, dtype, str(block), str(groups)],
        env=env,
        timeout=180,
        capture_output=True,
        text=True,
    )
    (directory / "stdout.txt").write_text(result.stdout)
    (directory / "stderr.txt").write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads(Path(str(out) + ".json").read_text())
    assert receipt["requested_route"] == route
    assert receipt["route"] == ("msl" if dtype == "fp32_to_fp16" else route)
    assert len(receipt["dispatches"]) == 1
    assert receipt["canaries"] == "intact"
    return np.load(out), receipt


@pytest.mark.parametrize(
    "dtype,block,groups",
    [("fp32", 128, 1), ("fp32", 128, 3), ("i32", 128, 3), ("fp32", 2048, 1), ("fp32_to_fp16", 256, 1)],
)
def test_elementwise_matches(tmp_path, dtype, block, groups):
    a, msl = _run(tmp_path / "msl", True, dtype, block, groups)
    b, cpp = _run(tmp_path / "cpp", False, dtype, block, groups)
    assert msl["dispatches"][0]["groups"] == cpp["dispatches"][0]["groups"] == [groups, 1, 1]
    assert cpp["extension_sha256"]
    if dtype != "fp32_to_fp16":
        assert msl["metallib_sha256"] != cpp["metallib_sha256"]
    assert a.dtype == b.dtype and a.shape == b.shape
    assert a.tobytes() == b.tobytes()

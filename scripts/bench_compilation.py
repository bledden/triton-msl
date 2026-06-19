#!/usr/bin/env python3
"""Benchmark C++ metallib vs MSL metallib compilation time.

Each kernel compilation runs in a separate subprocess to avoid
Triton's in-memory JIT cache contamination between runs.
"""
import os
import sys
import time
import shutil
import subprocess

CACHE_DIRS = [
    os.path.expanduser("~/.triton/cache"),
    os.path.expanduser("~/.cache/triton_msl"),
]

PYTHON = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       ".venv", "bin", "python")

KERNEL_SCRIPTS = {
    "vector_add": '''
import time, torch, triton, triton.language as tl
@triton.jit
def k(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask), mask=mask)
n = 1024; x = torch.randn(n); y = torch.randn(n); out = torch.zeros(n)
t0 = time.perf_counter()
k[(triton.cdiv(n, 256),)](x, y, out, n, BLOCK=256)
t1 = time.perf_counter()
print(f"{(t1-t0)*1000:.1f}")
''',
    "softmax": '''
import time, torch, triton, triton.language as tl
@triton.jit
def k(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=float('-inf'))
    mx = tl.max(x, axis=0)
    e = tl.exp(x - mx)
    tl.store(out_ptr + offs, e / tl.sum(e, axis=0), mask=mask)
n = 256; x = torch.randn(n); out = torch.zeros(n)
t0 = time.perf_counter()
k[(1,)](x, out, n, BLOCK=256)
t1 = time.perf_counter()
print(f"{(t1-t0)*1000:.1f}")
''',
    "chain": '''
import time, torch, triton, triton.language as tl
@triton.jit
def k(a_ptr, b_ptr, out_ptr, n, alpha, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, (a * alpha + b) * (a - b), mask=mask)
n = 1024; a = torch.randn(n); b = torch.randn(n); out = torch.zeros(n)
t0 = time.perf_counter()
k[(triton.cdiv(n, 256),)](a, b, out, n, 2.5, BLOCK=256)
t1 = time.perf_counter()
print(f"{(t1-t0)*1000:.1f}")
''',
}


def clear_caches():
    for d in CACHE_DIRS:
        if os.path.exists(d):
            shutil.rmtree(d, ignore_errors=True)


def run_one(kernel_name, use_cpp):
    """Run a single kernel compilation in a fresh subprocess."""
    clear_caches()
    env = dict(os.environ)
    if use_cpp:
        env["TRITON_MSL_USE_CPP"] = "1"
    else:
        env.pop("TRITON_MSL_USE_CPP", None)

    result = subprocess.run(
        [PYTHON, "-c", KERNEL_SCRIPTS[kernel_name]],
        capture_output=True, text=True, env=env, timeout=120,
    )
    if result.returncode != 0:
        print(f"  ERROR ({kernel_name}): {result.stderr[-200:]}", file=sys.stderr)
        return -1.0
    return float(result.stdout.strip())


def bench(path_name, use_cpp, runs=3):
    print(f"\n{path_name}:")
    results = {}
    for name in KERNEL_SCRIPTS:
        times = []
        for _ in range(runs):
            t = run_one(name, use_cpp)
            times.append(t)
        avg = sum(times) / len(times)
        results[name] = times
        print(f"  {name:12s}: {avg:7.1f}ms avg  ({', '.join(f'{t:.0f}' for t in times)})")
    return results


def main():
    print("C++ vs MSL Metallib Compilation Benchmark")
    print("=" * 50)
    print(f"Each kernel compiled 3 times in separate subprocesses")
    print(f"(all caches cleared between runs)")

    msl = bench("MSL Path (default)", False)
    cpp = bench("C++ Metallib Path", True)

    print(f"\nSpeedup (MSL avg / C++ avg):")
    for name in KERNEL_SCRIPTS:
        msl_avg = sum(msl[name]) / len(msl[name])
        cpp_avg = sum(cpp[name]) / len(cpp[name])
        if cpp_avg > 0 and msl_avg > 0:
            ratio = msl_avg / cpp_avg
            faster = "C++" if ratio > 1 else "MSL"
            print(f"  {name:12s}: {ratio:.2f}x ({faster} faster)")


if __name__ == "__main__":
    main()

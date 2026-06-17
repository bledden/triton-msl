# Contributing to triton-metal

## Development Setup

### Prerequisites
- Apple Silicon Mac (M1 or later)
- macOS 14 (Sonoma) or later
- Xcode Command Line Tools: `xcode-select --install`
- Python 3.10+

### Install

```bash
git clone https://github.com/bledden/triton-metal.git
cd triton-metal
pip install -e ".[dev]"

# Install Triton (required for @triton.jit)
pip install triton>=3.6.0
# If no macOS wheel is available, build from source:
# pip install git+https://github.com/triton-lang/triton.git

# Install PyTorch (for torch.compile tests)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install MLX (for MLX backend tests)
pip install mlx
```

## Running Tests

```bash
# All local/project tests (current count is reported in README.md)
pytest tests/ -v

# Specific test suites
pytest tests/test_torch_compile.py -v      # torch.compile (32 tests)
pytest tests/test_mlx_backend.py -v        # MLX backend (15 tests)
pytest tests/test_generic_lowerer.py -v    # Codegen unit tests
pytest tests/test_gpu_correctness.py -v    # GPU correctness

# Upstream Triton test_core.py — this command IS the source of truth for the
# conformance count (README.md / CHANGELOG.md quote its latest result, dated).
# It runs with `--device cpu` (torch references compute on CPU while the Metal
# backend compiles + runs the kernels on the GPU) because upstream test_core
# assumes CUDA. Re-run it to regenerate reports/upstream_test_core.json; do NOT
# hand-edit pass counts into the docs.
python scripts/run_upstream_tests.py --test-file test_core.py --timeout 1800  # ~14 min
```

## Running Benchmarks

```bash
python benchmarks/bench_all.py             # All native kernel benchmarks
python benchmarks/bench_copy_overhead.py   # Buffer copy analysis
python benchmarks/mlx_vs_pyobjc.py         # MLX vs PyObjC comparison
```

## Code Style

We use [ruff](https://docs.astral.sh/ruff/) for linting:

```bash
ruff check triton_metal/ tests/
ruff format triton_metal/ tests/
```

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the compilation pipeline and design decisions.

Key directories:
- `triton_metal/codegen/` — TTGIR → MSL compilation (generic_lowerer.py is the primary path)
- `triton_metal/backend/` — Triton backend integration (compiler.py, driver.py)
- `triton_metal/inductor/` — torch.compile integration
- `triton_metal/mlx/` — MLX zero-copy backend

## Pull Requests

1. Fork the repo and create a branch from `main`
2. Make your changes
3. Run `pytest tests/ -v` to ensure all tests pass
4. Run `ruff check` to ensure code style compliance
5. Open a PR with a clear description of the change

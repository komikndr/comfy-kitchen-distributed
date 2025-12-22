# Comfy Kitchen

Fast kernel library for Diffusion inference with multiple compute backends.

## Backend Capabilities Matrix

| Function                    | eager | cuda | triton |
|-----------------------------|-------|------|--------|
| `quantize_per_tensor_fp8`   | ✓     | ✓    | ✓      |
| `dequantize_per_tensor_fp8` | ✓     | ✓    | ✓      |
| `quantize_nvfp4`            | ✓     | ✓    | ✓      |
| `dequantize_nvfp4`          | ✓     | ✓    |        |
| `scaled_mm_nvfp4`           | ✓     | ✓    |        |
| `apply_rope`                | ✓     | ✓    | ✓      |
| `apply_rope1`               | ✓     | ✓    | ✓      |


## Quantized Tensors

The library provides `QuantizedTensor`, a `torch.Tensor` subclass that transparently intercepts PyTorch operations and dispatches them to optimized quantized kernels when available.

| Layout                 | Format       | HW Requirement  | Description                             |
|------------------------|--------------|-----------------|----------------------------------------|
| `TensorCoreFP8Layout`  | FP8 E4M3     | SM ≥ 8.9 (Ada)  | Per-tensor scaling, 1:1 element mapping |
| `TensorCoreNVFP4Layout`| NVFP4 E2M1   | SM ≥ 10.0 (Blackwell) | Block quantization with 16-element blocks |

```python
from comfy_kitchen.tensor import QuantizedTensor, TensorCoreFP8Layout, TensorCoreNVFP4Layout

# Quantize a tensor
x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
qt = QuantizedTensor.from_float(x, TensorCoreFP8Layout)

# Operations dispatch to optimized kernels automatically
output = torch.nn.functional.linear(qt, weight_qt)

# Dequantize back to float
dq = qt.dequantize()
```


## Installation

### Package Variants

Two package variants are available:

1. **`comfy-kitchen`** (default) - Full package with CUDA support
   - Requires CUDA toolkit for compilation
   - Single wheel for Python 3.12+ (Stable ABI)

2. **`comfy-kitchen-no-cuda`**
   - Pure Python wheel (`py3-none-any`)
   - Includes: eager and triton backends only
   - No compilation required, works on any platform


### Installation Options

```bash
# Standard installation with CUDA support
pip install .

# Development installation
pip install -e ".[dev]"

# For faster rebuilds during development:
# Skip build isolation for faster rebuilds
pip install -e . --no-build-isolation -v

```

#### Available Build Options

These options require using `setup.py` directly (not `pip install`):

| Option | Command | Description | Default |
|--------|---------|-------------|---------|
| `--no-cuda` | `python setup.py bdist_wheel --no-cuda` | Build without CUDA backend | Enabled (build with CUDA) |
| `--cuda-archs=...` | `python setup.py build_ext --cuda-archs="80;89"` | CUDA architectures to build for | Windows: `80;89;120f`<br>Linux: `80;89;90a;100f;120f` |
| `--debug-build` | `python setup.py build_ext --debug-build` | Build in debug mode with symbols | Disabled (Release) |
| `--lineinfo` | `python setup.py build_ext --lineinfo` | Enable NVCC line info for profiling | Disabled |

```bash
# Build without CUDA
python setup.py bdist_wheel --no-cuda

# Build with custom CUDA architectures
python setup.py build_ext --cuda-archs="80;89" bdist_wheel

# Debug build with line info for profiling
python setup.py build_ext --debug-build --lineinfo bdist_wheel
```



### Requirements

- **Python**: ≥3.12 (uses Stable ABI - single wheel works across 3.12, 3.13, 3.14+)
- **PyTorch**: ≥2.5.0
- **CUDA Toolkit** (optional): ≥12.8 for CUDA backend
  - Set `CUDA_HOME` environment variable if not auto-detected
- **nanobind**: ≥2.0.0 (for building CUDA extension)
- **CMake**: ≥3.18 (for building CUDA extension)

## Quick Start

```python
import comfy_kitchen as ck
import torch

# Automatic backend selection (triton -> cuda -> eager)
x = torch.randn(100, 100, device="cuda")
scale = torch.tensor([1.0], device="cuda")
result = ck.quantize_per_tensor_fp8(x, scale)

# Check which backends are available
print(ck.list_backends())

# Force a specific backend
result = ck.quantize_per_tensor_fp8(x, scale, backend="eager")

# Temporarily use a different backend
with ck.use_backend("cuda"):
    result = ck.quantize_per_tensor_fp8(x, scale)
```

## Backend System

The library supports multiple backends:
- **eager**: Pure PyTorch implementation
- **cuda**: Custom CUDA C kernels (CUDA only)
- **triton**: Triton JIT-compiled kernels

### Automatic Backend Selection

When you call a function, the registry selects the best backend by checking **constraints** in priority order (`cuda` → `triton` → `eager`):

```python
# Backend is selected automatically based on input constraints
result = ck.quantize_per_tensor_fp8(x, scale)

# On CPU tensors → falls back to eager (only backend supporting CPU)
# On CUDA tensors → uses cuda or triton (higher priority)
```

### Constraint System

Each backend declares constraints for its functions:

| Constraint | Description |
|------------|-------------|
| **Device** | Which device types are supported (`cuda`, `cpu`) |
| **Dtype** | Allowed input/output dtypes per parameter |
| **Shape** | Shape requirements (e.g., 2D tensors, dimensions divisible by 16) |
| **Compute Capability** | Minimum GPU architecture (e.g., SM 8.0 for FP8, SM 10.0 for NVFP4) |

The registry validates inputs against these constraints **before** calling the backend—no try/except fallback patterns. If no backend can handle the inputs, a `NoCapableBackendError` is raised with details.

```python
# Debug logging to see backend selection
import logging
logging.getLogger("comfy_kitchen.dispatch").setLevel(logging.DEBUG)
```


## Testing

Run the test suite with pytest:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_backends.py

# Run with verbose output
pytest -v

# Run specific test
pytest tests/test_backends.py::TestBackendSystem::test_list_backends
```

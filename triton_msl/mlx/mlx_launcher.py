"""MLX launcher for Triton-compiled Metal kernels.

Dispatches MSL kernels via mx.fast.metal_kernel() with zero-copy
MLX array binding, replacing the PyObjC Metal dispatch path.

MLX arrays are immutable — metal_kernel returns NEW output arrays.
The launcher collects shapes/dtypes from the output arg placeholders
passed by the caller.
"""

import mlx.core as mx

from triton_msl.mlx.msl_extractor import MSLExtraction


# Map MSL type names to MLX dtypes
_MSL_TO_MLX_DTYPE = {
    "float": mx.float32,
    "half": mx.float16,
    "bfloat": mx.bfloat16,
    "int": mx.int32,
    "uint": mx.uint32,
    "short": mx.int16,
    "ushort": mx.uint16,
    "char": mx.int8,
    "uchar": mx.uint8,
    "bool": mx.bool_,
}


def _mx_dtype_from_array(arr):
    """Get MLX dtype from an MLX array."""
    return arr.dtype


def _mx_dtype_str(dtype):
    """Convert MLX dtype to MSL type string for metal_kernel."""
    dtype_map = {
        mx.float32: "float",
        mx.float16: "half",
        mx.bfloat16: "bfloat",
        mx.int32: "int",
        mx.uint32: "uint",
        mx.int16: "short",
        mx.uint16: "ushort",
        mx.int8: "char",
        mx.uint8: "uchar",
        mx.bool_: "bool",
    }
    return dtype_map.get(dtype, "float")


class MLXLauncher:
    """Launch Triton-compiled MSL kernels via MLX's metal_kernel API.

    Args:
        extraction: MSLExtraction from extract_msl_for_mlx().
        block_size: Threads per threadgroup.
        needs_2d_grid: Whether the kernel uses multi-axis program_id.
    """

    def __init__(self, extraction: MSLExtraction, block_size: int = 256,
                 needs_2d_grid: bool = False):
        self.ext = extraction
        self.block_size = min(block_size, 1024)
        self.needs_2d_grid = needs_2d_grid
        self._kernel = None

    def _build_kernel(self):
        """Lazily build the mx.fast.metal_kernel."""
        self._kernel = mx.fast.metal_kernel(
            name=self.ext.kernel_name,
            input_names=self.ext.input_names + self.ext.scalar_names,
            output_names=self.ext.output_names,
            source=self.ext.body,
            header=self.ext.header,
            ensure_row_contiguous=True,
        )

    def __call__(self, grid, *args):
        """Dispatch the kernel with MLX arrays.

        Args:
            grid: Tuple of threadgroup counts (gridX,) or (gridX, gridY) or
                  (gridX, gridY, gridZ).
            *args: Kernel arguments in Triton signature order (excluding
                   constexpr args). Pointer args are MLX arrays (inputs)
                   or MLX arrays / None (outputs). Scalar args are Python
                   int/float values.

        Returns:
            List of output MLX arrays, in output_names order.
        """
        if self._kernel is None:
            self._build_kernel()

        ptr_count = len(self.ext.input_names) + len(self.ext.output_names)
        output_ptr_indices = self.ext.output_ptr_indices or set()

        input_arrays = []
        output_shapes = []
        output_dtypes = []

        ptr_idx = 0
        for arg in args:
            if ptr_idx < ptr_count:
                # This arg corresponds to a pointer parameter
                if ptr_idx in output_ptr_indices:
                    # Output pointer — extract shape/dtype for MLX
                    if isinstance(arg, mx.array):
                        output_shapes.append(arg.shape)
                        output_dtypes.append(arg.dtype)
                    elif arg is None:
                        raise ValueError(
                            f"Output arg at position {ptr_idx} is None. "
                            "Pass an MLX array (e.g. mx.zeros(...)) to specify "
                            "output shape and dtype."
                        )
                    else:
                        raise TypeError(
                            f"Output arg at position {ptr_idx} must be an MLX array, "
                            f"got {type(arg)}"
                        )
                else:
                    # Input pointer — pass to metal_kernel
                    input_arrays.append(arg)
                ptr_idx += 1
            else:
                # Scalar argument — pass as Python value in inputs
                if isinstance(arg, mx.array):
                    input_arrays.append(arg.item())
                else:
                    input_arrays.append(arg)

        # Inject tpg (threadgroups_per_grid) scalar values if needed.
        # These correspond to the __tpg_dimN scalar args added by the extractor.
        if self.ext.tpg_axes:
            grid_dims = {
                0: grid[0] if len(grid) > 0 else 1,
                1: grid[1] if len(grid) > 1 else 1,
                2: grid[2] if len(grid) > 2 else 1,
            }
            for axis in sorted(self.ext.tpg_axes):
                input_arrays.append(grid_dims[axis])

        # Compute Metal grid (total threads, not threadgroup counts)
        gridX = grid[0] if len(grid) > 0 else 1
        gridY = grid[1] if len(grid) > 1 else 1
        gridZ = grid[2] if len(grid) > 2 else 1

        if self.needs_2d_grid:
            metal_grid = (gridX * self.block_size, gridY, gridZ)
            threadgroup = (self.block_size, 1, 1)
        else:
            total_tg = gridX * gridY * gridZ
            metal_grid = (total_tg * self.block_size, 1, 1)
            threadgroup = (self.block_size, 1, 1)

        outputs = self._kernel(
            inputs=input_arrays,
            grid=metal_grid,
            threadgroup=threadgroup,
            output_shapes=output_shapes,
            output_dtypes=output_dtypes,
        )

        return outputs

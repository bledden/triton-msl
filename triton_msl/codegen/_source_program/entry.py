"""Emitter integration for a closed saved-LSE source program.

The envelope is deliberately broader than the proof. Once a seven-pointer,
four-scalar online-attention candidate enters it, a failed proof is terminal,
including with the legacy parser enabled. Neither names nor hashes recognize
programs. This package emits source but does not authorize public execution.
"""

from dataclasses import asdict
import hashlib

from . import precision, renderer, semantics, uniform

KIND = "source_program_pending_v1"


def candidate(graph):
    args = graph.args
    if len(args) != 11 or [a.is_ptr for a in args] != [True] * 7 + [False] * 4:
        return False
    names = {op.op for op in uniform.walk(graph.ops)}
    # Alternative markers keep single-operation near misses in the proof path.
    return "tt.dot" in names and bool(names & {"math.exp2", "math.log", "arith.maxnumf"})


def emit(module, graph, options, entry, lowerer_type):
    """Return source and packed metadata, or None for another program family."""
    from triton_msl.errors import MetalNonRecoverableError

    try:
        if not candidate(graph):
            return None
        from triton._C.libtriton import ir

        # The graph was walked from this live module by emit_msl. Native dot
        # identities, operands and attributes are checked against that SAME graph.
        program, receipt = precision.capture(module, ir, lambda mod, opts: graph, options, semantics, lowerer_type)
        enabled = options.experimental_source_program_cooperative
        if enabled:
            from . import cooperative

            source, manifest = cooperative.render(program, entry)
            threads, shared = manifest["threads"], manifest["shared_bytes"]
        else:
            source, manifest = renderer.Row(program).generate(entry)
            threads, shared = 32, 0
        descriptor = [
            KIND,
            {
                "schema": 1,
                "entry": entry,
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "branch": "padded" if program["padded"] else "masked",
                "bindings": 11,
                "outputs": [0, 1],
                "scalar_roles": [[7, "f32", "scale"], [8, "f32", "sentinel"], [9, "i32", "N"], [10, "i32", "H"]],
                "threads": [threads, 1, 1],
                "full_tile_domain": {
                    "N": [32, 64, 128, 256] if enabled else [32, 64],
                    "H": [1, 2],
                    "batch": [1, 2],
                    "DIM": 32,
                    "BLOCK": 32,
                },
                "addresses": [asdict(item["record"]) for item in program["addresses"]],
                "native_ieee_enum": receipt["ieee_enum"],
                "max_imprecise_accumulation": 0,
                "private_bytes_declared": manifest["declared_private_bytes_upper_bound"],
                "runtime_admission": "blocked: numerical, storage and dispatch qualification pending",
            },
        ]
        return source, {
            "block_size": threads,
            "shared": shared,
            "output_arg_indices": [0, 1],
            "needs_2d_grid": True,
            "mm_two_kernel": None,
            "fast_matmul": None,
            "quant_matmul": None,
            "flash_attention": None,
            "batched_dot_bounds": descriptor,
            "device_assert": None,
            "argument_abi": None,
        }
    except Exception as exc:
        raise MetalNonRecoverableError(
            "source-program proof refused; no legacy fallback: " + str(exc), op_name="tt.dot"
        ) from exc

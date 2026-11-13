"""Export a proved source-program TTGIR module without creating a GPU device.

The external consumer owns dispatch geometry, buffer bounds and synchronization.
This command does not enable the Python public launcher's pending admission route.
"""

import argparse
import hashlib
import json
from pathlib import Path


def export(ttgir, output, *, cooperative=False, metal_version="4.0", source_only=False):
    from triton._C.libtriton import ir
    from triton_msl.backend.compiler import MetalBackend, MetalOptions
    from triton_msl.codegen._source_program.entry import KIND

    ttgir, output = Path(ttgir), Path(output)
    if output.exists():
        raise FileExistsError("export destination already exists")
    context = ir.context()
    ir.load_dialects(context)
    module = ir.parse_mlir_module(str(ttgir), context)
    options = MetalOptions(
        target_metal_version=metal_version,
        experimental_source_program_cooperative=cooperative,
    )
    metadata = {}
    source = MetalBackend.make_msl(module, metadata, options)
    descriptor = metadata.get("batched_dot_bounds")
    if not descriptor or descriptor[0] != KIND:
        raise ValueError("not a proved source-program export")
    binary = None if source_only else MetalBackend.make_metallib(source, metadata, options)
    files = {"kernel.metal": source.encode("utf-8")}
    if binary is not None:
        files["kernel.metallib"] = binary
    manifest = {
        "schema": 1,
        "input_sha256": hashlib.sha256(ttgir.read_bytes()).hexdigest(),
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
        "metadata": metadata,
        "options": dict(options.__dict__),
        "scope": "external export; caller enforces bounds, geometry, lifetimes and synchronization",
        "public_dispatch_enabled": False,
    }
    # Return the same JSON-shaped contract that an external consumer reads.
    manifest = json.loads(json.dumps(manifest))
    output.mkdir(parents=True, exist_ok=False)
    for name, data in files.items():
        (output / name).write_bytes(data)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ttgir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    parser.add_argument("--cooperative", action="store_true", help="opt in to the 128-thread lowering")
    parser.add_argument("--metal-version", default="4.0", help="installed toolchain language version")
    parser.add_argument("--source-only", action="store_true", help="do not invoke the Metal compiler")
    args = parser.parse_args(argv)
    manifest = export(
        args.ttgir,
        args.output,
        cooperative=args.cooperative,
        metal_version=args.metal_version,
        source_only=args.source_only,
    )
    print(json.dumps({"output": str(args.output), "files": manifest["files"]}))


if __name__ == "__main__":
    main()

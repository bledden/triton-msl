"""Inspect a named macOS extension without importing it or creating a device.

This is a narrow non-embedding check, not full ABI or dependency qualification.
Pass the expected image digest from the artifact manifest, not a guessed path.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def check_load_commands(listing):
    lines = listing.splitlines()
    if not lines or not lines[0].endswith(":"):
        raise ValueError("missing otool image header")
    dependencies = []
    for line in lines[1:]:
        match = re.fullmatch(r"\s+(.+?) \(compatibility version [^,]+, current version [^)]+\)", line)
        if match is None:
            raise ValueError(f"unrecognized dependency row: {line!r}")
        dependency = match.group(1)
        if "python.framework" in dependency.lower() or Path(dependency).name.lower().startswith("libpython"):
            raise ValueError(f"extension embeds a Python runtime: {dependency}")
        dependencies.append(dependency)
    if not dependencies or not any(Path(name).name == "libtriton.so" for name in dependencies):
        raise ValueError("missing matched libtriton dependency")
    return dependencies


def inspect(image, expected_sha256):
    image = Path(image).resolve(strict=True)
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError("artifact hash mismatch before inspection")
    listing = subprocess.check_output(["/usr/bin/otool", "-L", str(image)], text=True)
    dependencies = check_load_commands(listing)
    if hashlib.sha256(image.read_bytes()).hexdigest() != digest:
        raise ValueError("artifact changed during inspection")
    return {"image": str(image), "sha256": digest, "dependencies": dependencies, "no_python_link": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(inspect(args.image, args.sha256), indent=2))


if __name__ == "__main__":
    main()

# applegpu decoder

Copied from this repository's retained third_party/applegpu source, whose provenance
records https://github.com/dougallj/applegpu at commit
797862eea0cee1f1eba74be3d0d02be4b3d2bd0d, vendored 2026-05-30.

The installed subset is applegpu.py, fma.py and srgb.py. The only decoder changes
are its two imports, made package-relative to avoid collisions with applications'
top-level modules. The algorithm, fma.py and srgb.py are unchanged. The repository
copy remains the import-delta reference; no newer upstream revision is claimed.

LICENSE contains Dougall Johnson's BSD-3-Clause notice. fma.py retains the Go
Authors' BSD-3-Clause notice in its header. Both notices ship in the distributions.

Used for best-effort static archive inspection only. M4/AGX2 coverage is partial
and explicitly reported. This is not an authoritative instruction-set model or
a GPU execution component. The command-line wrapper and upstream test/hardware
tools are not bundled.

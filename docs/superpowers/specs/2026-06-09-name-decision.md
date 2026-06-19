# Distribution name decision — 2026-06-09 (Phase 0 T7); SUPERSEDED 2026-06-19

PyPI `triton-metal` is taken (chenxingqiang, v3.3.0rc2, identically pitched,
upstream repo deleted) and close variants like `tritonmetal` are blocked by
PyPI's PEP 541 confusability check. `pip install triton-metal` installs the
competitor — a rename is forced.

**Original decision (2026-06-09): `triton-metal-backend`** — discoverable,
accurate (a Triton *backend*, not a fork), unsquatted at decision time; import
name was to stay `triton_metal`, only the PyPI distribution name differing.

**SUPERSEDED 2026-06-19:** shipped instead as **`triton-msl`** with a *full
rename of the import package* to **`triton_msl`** (MSL = Metal Shading Language,
which this backend emits) for install-name/import-name consistency. Claimed on
PyPI at first publish (roadmap #2). Note: the Apple-Metal GPU API terms (the
`metal` backend/device id, `Metal*` classes, `.metal`/MSL) are intentionally
unchanged — only the project/package identity moved off the `triton_metal` stem.

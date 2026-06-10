# Distribution name decision — 2026-06-09 (Phase 0 T7)

PyPI `triton-metal` is taken (chenxingqiang, v3.3.0rc2, identically pitched,
upstream repo deleted). `pip install triton-metal` installs the competitor —
rename is forced.

**Decision: `triton-metal-backend`.** Discoverable, accurate (a Triton
*backend*, not a fork), unsquatted at decision time. Import stays
`triton_metal`; only the PyPI distribution name differs.

Gated to Phase 5: claim the name on PyPI when publishing; update README
install lines then.

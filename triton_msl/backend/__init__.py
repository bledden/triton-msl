"""Triton Metal backend plugin.

Triton discovers backends via entry points. The entry point
`triton.backends: metal = triton_msl.backend` tells Triton to import
`triton_msl.backend.compiler` and `triton_msl.backend.driver`
directly — no re-exports needed here.
"""

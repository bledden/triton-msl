"""triton-metal: Metal (Apple Silicon) backend for OpenAI Triton."""

# Bump on ANY emitter/lowerer change: persistent caches at ~/.cache/triton_metal
# are keyed by TTGIR + options only; without this, codegen fixes silently
# replay stale compiled kernels after upgrade (Phase 0, audit debt #1).
CODEGEN_VERSION = "2026.06.17.2"

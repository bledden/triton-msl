# Retained native lowering witnesses

Opaque IR copied byte for byte from Fable's CPU-only adapter review of 2026-09-20:
`triton-validation-evidence/2026-09-19-0.4.0-start/fable-cpp-cfg-adapter/probe-v2-*/<case>/opaque.ll`.

- `cpp-volatile-build`: accepted helper SHA-256 `3d68ba91431027a8bb6a6ef1be0e32121dc09fa7f0d94d6946cfefc24bcfcd0d`.
- `cpp-cfg-native-build`: scalar-pipeline helper SHA-256 `f11345c32c4a2d5bfa5ccabde92127ae380b6ebcc7295c65f8681caf655469b5`.

AIR 2.8 / Metal 4.0. These fixtures exercise pre-adapter selection without a
native helper, compiler subprocess, or GPU. Two-pointer and chained-pointer
fallbacks must preserve the accepted IR exactly. Pointer-select is an existing
unsupported case on both helpers, not a capability claim. Native integration
tests separately reproduce six pairs with the runtime switch in one helper.

# Vendored: applegpu

Apple GPU ISA disassembler by Dougall Johnson — REFERENCES.md [11].

- Upstream: https://github.com/dougallj/applegpu
- Commit:   797862eea0cee1f1eba74be3d0d02be4b3d2bd0d (HEAD at vendor time)
- Vendored: 2026-05-30
- Files:    applegpu.py, disassemble.py (the decoder only; hwtest/assemble/
            compiler_explorer etc. omitted)

Used by triton_msl/profiling/disasm.py for BEST-EFFORT native-AGX
disassembly. applegpu targets the M1-era AGX ISA; the M4 is AGX2, so decode
coverage on M4 is partial (reported explicitly by the harness). See
docs/INSTRUMENTS.md for the full-fidelity Xcode/Instruments path.

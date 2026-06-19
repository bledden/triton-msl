# Citing triton-msl

If you use `triton-msl` in research or technical work, please cite the
project alongside the underlying Triton compiler ([REFERENCES.md][1]).

## BibTeX

```bibtex
@software{triton_msl,
  title   = {triton-msl: Apple Silicon Metal backend for OpenAI Triton},
  author  = {Ledden, Blake},
  year    = {2026},
  url     = {https://github.com/bledden/triton-msl},
  version = {0.1.0-alpha},
  note    = {Apple Silicon Metal backend for the OpenAI Triton compiler;
             alpha release. See CHANGELOG.md for current status.}
}
```

Please also cite the original Triton paper:

```bibtex
@inproceedings{tillet2019triton,
  title     = {Triton: An Intermediate Language and Compiler for Tiled
               Neural Network Computations},
  author    = {Tillet, Philippe and Kung, H.T. and Cox, David},
  booktitle = {Proceedings of the 3rd ACM SIGPLAN International Workshop
               on Machine Learning and Programming Languages (MAPL '19)},
  pages     = {10--19},
  year      = {2019},
  doi       = {10.1145/3315508.3329973}
}
```

If your work depends on the FlashAttention path, the online-softmax trick,
MLX integration, the Asahi/`applegpu` disassembly tooling, or any other
external work, cite the corresponding entry from [`REFERENCES.md`](REFERENCES.md)
as well.

[1]: REFERENCES.md

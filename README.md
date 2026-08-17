# Rethinking Amortized Neural Representations for High-Resolution Terrain Elevation Data

**Haoan Feng, Xin Xu, Leila De Floriani** — University of Maryland, College Park

ACM SIGSPATIAL 2026

[**Project page**](https://fengyee.github.io/huvr_siren/) · [**arXiv**](https://arxiv.org/abs/2606.00404) · [**Paper (PDF)**](https://fengyee.github.io/huvr_siren/static/huvr_siren_sigspatial2026.pdf)

## TL;DR

Amortized neural representations encode each terrain tile as a compact per-tile
token that a shared coordinate decoder turns back into a continuous elevation
surface. We benchmark three representative methods on 1 m/pixel bare-earth
terrain under one protocol, and introduce **HUVR+SIREN**, which replaces the
strongest method's piecewise-linear coordinate decoder with a smooth,
analytically differentiable one: **+2.83 dB** at unchanged per-tile storage,
better gradient and Laplacian fidelity, and only 0.15 dB lost to int8
post-training quantization of the stored token.

## Code

**Code release coming soon.**

## Citation

```bibtex
@inproceedings{feng2026rethinking,
  title     = {Rethinking Amortized Neural Representations for High-Resolution Terrain Elevation Data},
  author    = {Feng, Haoan and Xu, Xin and De Floriani, Leila},
  booktitle = {Proceedings of the 34th ACM International Conference on Advances in Geographic Information Systems (SIGSPATIAL '26)},
  year      = {2026},
  eprint    = {2606.00404},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}
```

## License

Three different terms apply to the contents of this repository:

- **Site code** (`index.html`, `static/css/index.css`): MIT, see `LICENSE`.
- **Page template**: adapted from the
  [Nerfies project page](https://github.com/nerfies/nerfies.github.io),
  released under
  [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
- **Paper PDF and figures** (`static/huvr_siren_sigspatial2026.pdf`,
  `static/images/*`): © 2026 the authors; publication rights licensed to ACM.
  Not covered by the MIT license above — reuse is governed by the ACM
  copyright notice on the first page of the paper.

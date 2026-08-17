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

MIT (see `LICENSE`).

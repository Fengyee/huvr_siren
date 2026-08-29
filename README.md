# Rethinking Amortized Neural Representations for High-Resolution Terrain Elevation Data

**Haoan Feng, Xin Xu, Leila De Floriani** — University of Maryland, College Park

ACM SIGSPATIAL 2026

[**Project page**](https://fengyee.github.io/huvr_siren/) · [**Paper (arXiv)**](https://arxiv.org/abs/2606.00404)

![Pipeline diagram comparing TransINR, Functa, HUVR and HUVR+SIREN. A DEM tile is patchified, then routed either through a meta-learner whose per-tile latent is optimized at test time or through a hypernetwork encoder and decoder around a row of bottleneck tokens. Colored arrows carry each method into either a SIREN or a ReLU-MLP coordinate decoder mapping x and y to elevation, and patch-wise paths pass through patch stitching or pixelshuffle upsampling before the reconstructed tile.](static/images/teaser.png)

The four benchmarked methods differ in how the per-tile token is produced and
how it modulates the coordinate decoder.

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

Training code for both models is in [`code/`](code/) — see
[`code/README.md`](code/README.md) for install, a data-free smoke test, the
shard format, and the config tables.

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
- **Training code** (`code/`): MIT, with portions derived from
  [trans-inr](https://github.com/yinboc/trans-inr) (BSD 3-Clause) and
  [HUVR](https://github.com/tiktok/huvr) (MIT), as marked in the file headers.
  See [`code/README.md`](code/README.md#license).
- **Page template**: adapted from the
  [Nerfies project page](https://github.com/nerfies/nerfies.github.io),
  released under
  [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
- **Paper figures** (`static/images/*`): © 2026 the authors; publication
  rights licensed to ACM. Not covered by the MIT license above — reuse is
  governed by the ACM copyright notice in the published paper.

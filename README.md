<!-- PROJECT LOGO -->
<br />
<p align="center">

  <h1 align="center"><a href="">Transient Mitsuba 3</a></h1>

  <!-- <a href="">
    <img src="https://mcrespo.me/publications/primary-space-cv/figures/socialMedia.png" alt="Logo" width="100%">
  </a> -->

  <p align="center">
    <a href="https://mcrespo.me"><strong>Miguel Crespo</strong></a>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://diego.contact"><strong>Diego Royo</strong></a>
  </p>

  <!-- <p align="center">
    <a href='' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Docs-passing-green?style=flat-square' alt='Project Page'>
    </a>
    <a href='' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat-square' alt='Project Page'>
    </a>
  </p> -->
</p>

<br />
<br />

# Introduction

This library adds support to [Mitsuba 3](https://github.com/mitsuba-renderer/mitsuba3) for doing transient simulations. <!-- It works as an standalone python library, avoiding the need to compile the full system if you do not need anything custom. -->

Usual rendering is referred as steady in terms of the light has infinite propagation speed. In contrast, transient rendering lift this limitation allowing to simulate light in motion (see the teaser image for a visual example).

*TODO : improve explanation transient rendering.*

*TODO : put a video of the resulting simulation.*

*TODO : improve documentation.*

# Main features

* **Cross-platform:** Mitsuba 3 has been tested on Linux (x86_64), macOS (aarch64, x86_64), and Windows (x86_64).
<!-- * **Python only** library for doing transient rendering in both CPU and GPU. -->
* **Easy interface** to convert your algorithms for the transient domain.
* **Several integrators already implemented** including path tracing and volumetric path-tracing.
* **Temporal domain** filtering.

### Future / missing features

- [ ] Importance sampling of the temporal domain.
- [ ] Differentiation of the transient simulation.
- [ ] Fluorescence BRDF.
- [X] Non-line-of-sight support (NLOS)

# Installation

<!-- We provide the package via PyPI. Installing Mitsuba 3 transient this way is as simple as running

```bash
pip install mitransient
``` -->

_NOTE: These instructions have been tested on Linux only_

After cloning the repo, navigate to the root folder and execute the following commands to build the custom version of Mitsuba 3

```bash
git submodule update --init --recursive
cd ext/mitsuba3
mkdir -p build
cd build
cmake -GNinja ..
# Here, edit build/mitsuba.conf and set the enabled variants
# Recommended: scalar_rgb, llvm_rgb and cuda_rgb (FIXME: change to llvm_mono for NLOS?)
ninja
```

After this you are all set to use our transient version of Mitsuba 3

For NLOS simulations, see https://github.com/diegoroyo/tal

# Requirements

- `Python >= 3.8`
- `Mitsuba3 == 3.3.0` (included in this repo)
- (optional) For computation on the GPU: `Nvidia driver >= 495.89`
- (optional) For vectorized / parallel computation on the CPU: `LLVM >= 11.1`

# Usage

Here is a simple "Hello World" example that shows how simple it is to render a scene using Mitsuba 3 transient from Python:

```python
# Import the library using the alias "mi"
import mitsuba as mi
# Set the variant of the renderer
mi.set_variant('scalar_rgb')
# Import the package
import MiTransient as mitr
# Load a scene
scene = mi.load_dict(mitr.cornell_box())
# Prepare transient storage
transient_integrator = scene.integrator()
transient_integrator.prepare_transient(scene, 0)
# Render the scene and develop the data
data_steady, data_transient = mi.render(scene)
# Use the resulting tensor (steady: [PixelX, PixelY, Channels], transient: [PixelX, PixelY, TimeBins, Channels]) as you need, where steady is the sum over the temporal axis
...
```

# About

This project was created by [Miguel Crespo](https://mcrespo.me).

When using Mitsuba 3 transient in academic projects, please cite:

```bibtex
@software{crespo2022mitsuba3transient,
    title = {Mitsuba 3 transient renderer},
    author = {Miguel Crespo},
    note = {https://github.com/mcrescas/mitsuba3-transient},
    version = {0.0.0},
    year = 2022,
}
```

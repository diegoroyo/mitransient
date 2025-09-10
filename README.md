<div align="center">
<img align="center" src="https://github.com/mitsuba-renderer/mitsuba2/raw/master/docs/images/logo_plain.png" width="90" height="90"/>
</div>

<!-- PROJECT LOGO -->
<p align="center">
  <h1 align="center">mitransient</h1>
  <h3 align="center">Transient light transport in Mitsuba 3
  <br><br>
<a href='https://mitransient.readthedocs.io/en/latest/?badge=latest'>
      <img src='https://readthedocs.org/projects/mitransient/badge/?version=latest' alt='Documentation Status' /><a href='https://pypi.org/project/mitransient/'>
      <img src='https://img.shields.io/pypi/v/mitransient.svg?color=green' alt='PyPI version' />
  </a></h3>
</p>

<div align="center">
  <img src="https://raw.githubusercontent.com/diegoroyo/mitransient/main/.images/cornell-box.png" width="200" height="200"/>
  <img src="https://raw.githubusercontent.com/diegoroyo/mitransient/main/.images/cornell-box.gif" width="200" height="200"/>
  <img src="https://raw.githubusercontent.com/diegoroyo/mitransient/main/.images/nlos-Z.png" width="200" height="200"/>
  <img src="https://raw.githubusercontent.com/diegoroyo/mitransient/main/.images/nlos-Z.gif" width="200" height="200"/>
  <br>
  <img src="https://raw.githubusercontent.com/diegoroyo/mitransient/main/.images/polarization.gif" width="320" height="240"/>
  <img src="https://raw.githubusercontent.com/diegoroyo/mitransient/main/.images/staircase_steady.png" width="160" height="240"/>
  <img src="https://raw.githubusercontent.com/diegoroyo/mitransient/main/.images/staircase_transient.gif" width="160" height="240"/>
  <img src="https://raw.githubusercontent.com/diegoroyo/mitransient/main/.images/staircase_diff.gif" width="160" height="240"/>
</div>

<br />

# Overview

*mitransient* is a library adds support to [Mitsuba 3](https://github.com/mitsuba-renderer/mitsuba3) for doing transient simulations, with amazing support for non-line-of-sight (NLOS) data capture simulations, polarization tracking and differentiable transient rendering.

<br>

> [!TIP]
> **Check out our <a href="https://mitransient.readthedocs.io">online documentation (mitransient.readthedocs.io)</a> and our code examples:** <br>
> * [Quickstart: your first time-resolved render](https://github.com/diegoroyo/mitransient/blob/main/examples/transient/render_cbox_diffuse.ipynb)
> * [Overview of all examples](https://github.com/diegoroyo/mitransient/tree/main/examples)
> * [Non-line-of-sight (NLOS) transient rendering](https://github.com/diegoroyo/mitransient/blob/main/examples/transient-nlos/mitsuba3-transient-nlos.ipynb)
> * [Time-resolved polarization simulations](https://github.com/diegoroyo/mitransient/tree/main/examples/polarization)

### Main features
* **Foundation ready to use:** easy interface to convert your algorithms to the transient domain.
* **Python-only** library for doing transient rendering in both CPU and GPU.
* **Several integrators already implemented:** transient pathtracing (also adapted for NLOS scenes) and transient volumetric pathtracing.
* **Cross-platform:** Mitsuba 3 has been tested on Linux (x86_64), macOS (aarch64, x86_64), and Windows (x86_64).
* **Polarization tracking**
* **Differentiable transient rendering**

<br>

# License and citation

This project was started by [Diego Royo](https://diego.contact), [Miguel Crespo](https://mcrespo.me) and [Jorge Garcia-Pueyo](https://jgarciapueyo.github.io/). See below for the full list of `mitransient` contributors. Also see the [original Mitsuba 3 license and contributors](https://github.com/mitsuba-renderer/mitsuba3).

If you use our code in your project, please consider citing us using the following:

```bibtex
@misc{mitransient,
  title        = {mitransient},
  author       = {Royo, Diego and Crespo, Miguel and Garcia-Pueyo, Jorge},
  year         = 2024,
  journal      = {GitHub repository},
  doi          = {https://doi.org/10.5281/zenodo.11032518},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/diegoroyo/mitransient}}
}
```

Additionally, the NLOS features were implemented from our publication [Non-line-of-sight transient rendering](https://doi.org/10.1016/j.cag.2022.07.003). Please also consider citing us if you use them:

```bibtex
@article{royo2022non,
	title        = {Non-line-of-sight transient rendering},
	author       = {Diego Royo and Jorge García and Adolfo Muñoz and Adrian Jarabo},
	year         = 2022,
	journal      = {Computers & Graphics},
	doi          = {https://doi.org/10.1016/j.cag.2022.07.003},
	issn         = {0097-8493},
	url          = {https://www.sciencedirect.com/science/article/pii/S0097849322001200}
}
```

# What is transient rendering?

Conventional rendering is referred to as steady state, where the light propagation speed is assumed to be infinite. In contrast, transient rendering breaks this assumption allowing us to simulate light in motion (see the teaser image for a visual example).

For example, path tracing algorithms integrate over multiple paths that connect a light source with the camera. For a known path, transient path tracing uses the *very complex* formula of `time = distance / speed` (see [[Two New Sciences by Galileo]](https://en.wikipedia.org/wiki/Two_New_Sciences)) to compute the `time` when each photon arrives at the camera from the path's `distance` and light's `speed`. This adds a new `time` dimension to the captured images (i.e. it's a video now). The simulations now take new parameters as input: when to start recording the video, how long is each time step (framerate), and how many frames to record.

*Note: note that the `time` values we need to compute are very small (e.g. light takes only ~3.33 * 10^-9 seconds to travel 1 meter), `time` is usually measured in optical path distance. See [Wikipedia](https://en.wikipedia.org/wiki/Optical_path_length) for more information. TL;DR `opl = distance * refractive_index`*

# Installation

We provide the package via PyPI. To install `mitransient` you need to run:

```bash
pip install mitransient
```

which will also install the `mitsuba` Python package as a dependency.

> [!IMPORTANT]
> `mitransient` and `mitsuba` have different *variants* that specify the number of channels (RGB image, monochromatic, etc.), hardware acceleration (execution in CPU, GPU, etc.). If you install `mitransient`/`mitsuba` via `pip`, you will have access to [the following variants specified in this website](https://mitsuba.readthedocs.io/en/stable/src/key_topics/variants.html). There are more variants available, but you will have to compile Mitsuba 3 yourself.

> [!TIP]
> If you wish to use your own compiled Mitsuba 3, see the section below "If you use your own Mitsuba 3".

## Requirements

- `Python >= 3.8`
- `mitsuba >= 3.6.0`
- (optional) For computation on the GPU: `Nvidia driver >= 495.89`
- (optional) For vectorized / parallel computation on the CPU: `LLVM >= 11.1`

## After installation

At this point, you should be able to `import mitsuba` and `import mitransient` in your Python code (careful about setting the correct `PATH` environment variable if you have compiled Mitsuba 3 yourself, see the section below).

For NLOS data capture simulations, see https://github.com/diegoroyo/tal. `tal` is a toolkit that allows you to create and simulate NLOS scenes with an easier shell interface instead of directly from Python.

### If you use your own Mitsuba 3

First, you will want to install `mitransient` without the `mitsuba` dependency:

```
pip install mitransient --no-dependencies
```

Then you will need to `import mitsuba` in your Python scripts. Concretely, the `PYTHONPATH` variable should point to the Mitsuba module that is built upon compilation. There are different ways to do so:

* One solution is to directly execute `setpath.sh` provided after the compilation of the Mitsuba 3 repo [(More info)](https://mitsuba.readthedocs.io/en/latest/src/developer_guide/compiling.html). This shell script will modify the `PATH` and `PYTHONPATH` variables to load first this version of Mitsuba.
* Another solution following the previous one is to directly set yourself the `PYTHONPATH` environment variable as you wish.
* Another solution for having a custom version globally available is by using `pip install . --editable`. This will create a symlink copy of the package files inside the corresponding `site-packages` folder and will be listed as a package installed of `pip` and will be available as other packages installed. If you recompile them, you will still have the newest version directly to use. Please follow these instructions:
  * Go to `<mitsuba-path>/mitsuba3/build/python/drjit` and execute `pip install . --editable`.
  * Go to `<mitsuba-path>/mitsuba3/build/python/mitsuba` and execute `pip install . --editable`.
* If you are a user of Jupyter Notebooks, the easiest solution will be to add the following snippet of code to modify the notebook's `PYTHONPATH`:
```python
import sys
sys.path.insert(0, '<mitsuba-path>/mitsuba3/build/python')
import mitsuba as mi
```

# Usage

> [!TIP]
> **Check out the `examples` folder for practical usage!** <br>

You are now prepared to render your first transient scene with mitransient. Running the code below will render the famous Cornell Box scene in transient domain and show a video.

```python
import mitsuba as mi
mi.set_variant('llvm_ad_rgb')
import mitransient as mitr

scene = mi.load_dict(mitr.cornell_box())
transient_integrator = scene.integrator()
transient_integrator.prepare_transient(scene, sensor=0)
img_steady, img_transient = transient_integrator.render(scene, spp=1024)

import numpy as np
img_transient_tonemap = mitr.vis.tonemap_transient(
    np.moveaxis(img_transient, 0, 1)
)
mitr.vis.show_video(
      img_transient_tonemap,
      axis_video=2,
)
```

## Plugins implemented

> [!TIP]
> You can also look at the plugins' documentation in [our online documentation](https://mitransient.readthedocs.io). Look on the left menu for Integrators, Films, Emitters and Sensors.

`mitransient` implements the following plugins which can be used in scene XML files and dictionaries. To view a description of their parameters, click on the name of your desired plugin.
* `film`:
  * [`transient_hdr_film`](https://github.com/diegoroyo/mitransient/blob/main/mitransient/films/transient_hdr_film.py): Transient equivalent of Mitsuba 3's `hdrfilm` plugin.
  * [`phasor_hdr_film`](https://github.com/diegoroyo/mitransient/blob/main/mitransient/films/phasor_hdr_film.py): Similar to `transient_hdr_film`, but stores the result in the frequency domain instead of the time domain.
* `integrators`:
  * [`transient_path`](https://github.com/diegoroyo/mitransient/blob/main/mitransient/integrators/transientpath.py): Transient path tracing for line-of-sight scenes. If you want to do NLOS simulations, use `transientnlospath` instead.
  * [`transient_nlos_path`](https://github.com/diegoroyo/mitransient/blob/main/mitransient/integrators/transientnlospath.py): Transient path tracing with specific sampling routines for NLOS scenes (e.g. laser sampling and hidden geometry sampling of the ["Non-Line-of-Sight Transient Rendering" paper](https://diego.contact/publications/nlos-render)).
  * [`transient_prbvolpath`](https://github.com/diegoroyo/mitransient/blob/main/mitransient/integrators/transient_prbvolpath.py): Path Replay Backpropagation for time-resolved volumetric path tracing.
* `emitters`:
  * [`angulararea`](https://github.com/diegoroyo/mitransient/blob/main/mitransient/emitters/angulararea.py): Similar to an `area` emitter, but emits within a restricted angular range. 
* `sensors`:
  * [`nlos_capture_meter`](https://github.com/diegoroyo/mitransient/blob/main/mitransient/sensors/nloscapturemeter.py): Can be attached to one of the scene's geometries, and measures uniformly-spaced points on such geometry (e.g. relay wall).

## Other useful functions

See the full list [on this website](https://mitransient.readthedocs.io/en/latest/src/other.html)

## Testing

Our test suite can be run using `pytest` on the root folder of the repo.

# Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://diego.contact/"><img src="https://avatars.githubusercontent.com/u/17049331?v=4?s=100" width="100px;" alt="Diego Royo"/><br /><sub><b>Diego Royo</b></sub></a><br /><a href="#design-diegoroyo" title="Design">🎨</a> <a href="https://github.com/diegoroyo/mitransient/commits?author=diegoroyo" title="Code">💻</a> <a href="#promotion-diegoroyo" title="Promotion">📣</a> <a href="https://github.com/diegoroyo/mitransient/commits?author=diegoroyo" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/mcrescas"><img src="https://avatars.githubusercontent.com/u/62649574?v=4?s=100" width="100px;" alt="Miguel Crespo"/><br /><sub><b>Miguel Crespo</b></sub></a><br /><a href="#design-mcrescas" title="Design">🎨</a> <a href="https://github.com/diegoroyo/mitransient/commits?author=mcrescas" title="Code">💻</a> <a href="#promotion-mcrescas" title="Promotion">📣</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://jgarciapueyo.github.io/"><img src="https://avatars.githubusercontent.com/u/35239547?v=4?s=100" width="100px;" alt="Jorge Garcia Pueyo"/><br /><sub><b>Jorge Garcia Pueyo</b></sub></a><br /><a href="#design-jgarciapueyo" title="Design">🎨</a> <a href="https://github.com/diegoroyo/mitransient/commits?author=jgarciapueyo" title="Code">💻</a> <a href="#promotion-jgarciapueyo" title="Promotion">📣</a> <a href="https://github.com/diegoroyo/mitransient/commits?author=jgarciapueyo" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/DiegoBielsa"><img src="https://avatars.githubusercontent.com/u/71701253?v=4?s=100" width="100px;" alt="DiegoBielsa"/><br /><sub><b>DiegoBielsa</b></sub></a><br /><a href="https://github.com/diegoroyo/mitransient/issues?q=author%3ADiegoBielsa" title="Bug reports">🐛</a> <a href="https://github.com/diegoroyo/mitransient/commits?author=DiegoBielsa" title="Code">💻</a> <a href="https://github.com/diegoroyo/mitransient/commits?author=DiegoBielsa" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/p-luesia"><img src="https://avatars.githubusercontent.com/u/93714843?v=4?s=100" width="100px;" alt="p-luesia"/><br /><sub><b>p-luesia</b></sub></a><br /><a href="https://github.com/diegoroyo/mitransient/commits?author=p-luesia" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://opueyociutad.github.io/"><img src="https://avatars.githubusercontent.com/u/24795695?v=4?s=100" width="100px;" alt="Óscar Pueyo Ciutad"/><br /><sub><b>Óscar Pueyo Ciutad</b></sub></a><br /><a href="https://github.com/diegoroyo/mitransient/commits?author=opueyociutad" title="Code">💻</a> <a href="https://github.com/diegoroyo/mitransient/commits?author=opueyociutad" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Guilleuz"><img src="https://avatars.githubusercontent.com/u/90970380?v=4?s=100" width="100px;" alt="Guillermo Enguita Lahoz"/><br /><sub><b>Guillermo Enguita Lahoz</b></sub></a><br /><a href="https://github.com/diegoroyo/mitransient/commits?author=Guilleuz" title="Code">💻</a> <a href="https://github.com/diegoroyo/mitransient/commits?author=Guilleuz" title="Documentation">📖</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/lyd405121"><img src="https://avatars.githubusercontent.com/u/16344694?v=4?s=100" width="100px;" alt="蕉太狼"/><br /><sub><b>蕉太狼</b></sub></a><br /><a href="https://github.com/diegoroyo/mitransient/issues?q=author%3Alyd405121" title="Bug reports">🐛</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

See [emoji key](https://allcontributors.org/docs/en/emoji-key) for details.
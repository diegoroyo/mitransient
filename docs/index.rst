.. image:: ../.images/cornell-box.png
      :width: 23%
.. image:: ../.images/cornell-box.gif
      :width: 23%
.. image:: ../.images/nlos-Z.png
      :width: 23%
.. image:: ../.images/nlos-Z.gif
      :width: 23%

Getting started
===============

``mitransient`` is a library adds support to Mitsuba 3 for doing transient simulations, with amazing support for non-line-of-sight (NLOS) data capture simulations.

Main features
-------------

* **Foundation ready to use:** easy interface to convert your algorithms to the transient domain.
* **Python-only** library for doing transient rendering in both CPU and GPU.
* **Several integrators already implemented:** *transient pathtracing*  (also adapted for NLOS scenes) and *transient volumetric pathtracing*.
* **Cross-platform:** Mitsuba 3 has been tested on Linux (x86_64), macOS (aarch64, x86_64), and Windows (x86_64).
* **Temporal domain** filtering.

The following video showcases the potential application of ``mitransient`` to simulate light at a trillion frames per second, imitating a femto-photography experiment.

..  youtube:: wZfS19i6qkA
      :align: center
      :privacy_mode:

Installation
------------

We provide the package via PyPI. To install ``mitransient`` you need to run:

.. code-block:: bash

      pip install mitransient

If you have installed Mitsuba 3 via ``pip`` you will only have access to the ``llvm_ad_rgb`` and ``cuda_ad_rgb`` variants. If you want to use other variants (e.g. NLOS simulations can greatly benefit from the ``llvm_mono`` variant which only propagates one wavelength), then we recommend that you compile Mitsuba 3 yourself `following this tutorial <https://mitsuba.readthedocs.io/en/latest/src/developer_guide/compiling.html>`_ and enable the following variants: ``["scalar_mono", "llvm_mono", "llvm_ad_mono", "cuda_mono", "cuda_ad_mono", "scalar_rgb", "llvm_rgb", "llvm_ad_rgb", "cuda_rgb", "cuda_ad_rgb"]``.
For more information about requirements or using a custom Mitsuba 3 compilation, see :doc:`src/installation`.

Usage
-----

You should be able to render your first transient scene with ``mitransient``. Running the code below will render the famous Cornell Box scene in transient domain and show a video.

.. code-block:: python

      import mitsuba as mi
      mi.set_variant('scalar_rgb')
      import mitransient as mitr

      scene = mi.load_dict(mitr.cornell_box())
      transient_integrator = scene.integrator()
      transient_integrator.prepare_transient(scene, sensor=0)
      img_steady, img_transient = transient_integrator.render(scene)

      mitr.utils.show_video(
            np.moveaxis(img_transient, 0, 1),
            axis_video=2,
      )

License and citation
--------------------

This project was created by `Miguel Crespo <https://mcrespo.me>`_ and expanded by `Diego Royo <https://diego.contact>`_ and `Jorge García-Pueyo <https://jgarciapueyo.github.io/>`_. Also see the `original Mitsuba 3 license and contributors <https://github.com/mitsuba-renderer/mitsuba3>`_.
If you use our code in your project, please consider citing us using the following:

.. code-block:: bibtex

      @misc{mitransient,
            title        = {mitransient},
            author       = {Royo, Diego and Crespo, Miguel and Garcia-Pueyo, Jorge},
            year         = 2023,
            journal      = {GitHub repository},
            doi          = {https://doi.org/10.5281/zenodo.11032518},
            publisher    = {GitHub},
            howpublished = {\url{https://github.com/diegoroyo/mitransient}}
      }

Additionally, the NLOS features were re-implemented from our publication `Non-line-of-sight transient rendering <https://doi.org/10.1016/j.cag.2022.07.003>`_. Please also consider citing us if you use them:

.. code-block:: bibtex

      @article{royo2022non,
            title        = {Non-line-of-sight transient rendering},
            author       = {Diego Royo and Jorge García and Adolfo Muñoz and Adrian Jarabo},
            year         = 2022,
            journal      = {Computers & Graphics},
            doi          = {https://doi.org/10.1016/j.cag.2022.07.003},
            issn         = {0097-8493},
            url          = {https://www.sciencedirect.com/science/article/pii/S0097849322001200}
      }

What is transient rendering?
----------------------------

Conventional rendering is referred to as steady state, where the light propagation speed is assumed to be infinite. In contrast, transient rendering breaks this assumption allowing us to simulate light in motion (see the teaser image for a visual example).

For example, path tracing algorithms integrate over multiple paths that connect a light source with the camera. For a known path, transient path tracing uses the very complex formula of time = distance / speed (see [Two New Sciences by Galileo]) to compute the time when each photon arrives at the camera from the path's distance and light's speed. This adds a new time dimension to the captured images (i.e. it's a video now). The simulations now take new parameters as input: when to start recording the video, how long is each time step (framerate), and how many frames to record.

Note: note that the time values we need to compute are very small (e.g. light takes only ~3.33 * 10^-9 seconds to travel 1 meter), time is usually measured in optical path distance. See Wikipedia for more information. TL;DR opl = distance * refractive_index

.. .....................................................
.. toctree::
      :hidden:
      
      self
      src/installation

.. toctree::
      :maxdepth: 1
      :caption: Tutorials
      :hidden:
      
      src/transient_rendering_tutorials
      src/nlos_tutorials

.. toctree::
      :maxdepth: 1
      :caption: References
      :hidden:

      generated/plugin_reference/section_integrators
      generated/plugin_reference/section_films
      generated/plugin_reference/section_sensors
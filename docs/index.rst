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

This library adds support to Mitsuba 3 for doing transient simulations, with amazing support for non-line-of-sight (NLOS) data capture simulations.

Main features
-------------

* Foundation ready to use: easy interface to convert your algorithms to the transient domain.
* Python-only library for doing transient rendering in both CPU and GPU.
* Several integrators have already been implemented including *transient pathtracing*  (also adapted for NLOS scenes) and *volumetric pathtracing*.
* Cross-platform: Mitsuba 3 has been tested on Linux (x86_64), macOS (aarch64, x86_64), and Windows (x86_64).
* Temporal domain filtering.

What is transient rendering?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Conventional rendering is referred to as steady state, where the light propagation speed is assumed to be infinite. In contrast, transient rendering breaks this assumption allowing us to simulate light in motion (see the teaser image for a visual example).

For example, path tracing algorithms integrate over multiple paths that connect a light source with the camera. For a known path, transient path tracing uses the very complex formula of time = distance / speed (see [Two New Sciences by Galileo]) to compute the time when each photon arrives at the camera from the path's distance and light's speed. This adds a new time dimension to the captured images (i.e. it's a video now). The simulations now take new parameters as input: when to start recording the video, how long is each time step (framerate), and how many frames to record.

Note: note that the time values we need to compute are very small (e.g. light takes only ~3.33 * 10^-9 seconds to travel 1 meter), time is usually measured in optical path distance. See Wikipedia for more information. TL;DR opl = distance * refractive_index

Installation
------------
TBD

.. .....................................................
.. toctree::
      :hidden:
      
      self

.. toctree::
      :maxdepth: 1
      :caption: Tutorials
      :hidden:
      
      src/tutorials_transient
      src/tutorials_nlos

.. toctree::
      :maxdepth: 1
      :caption: References
      :hidden:

      generated/plugin_reference/section_integrators
      generated/plugin_reference/section_films
      generated/plugin_reference/section_sensors
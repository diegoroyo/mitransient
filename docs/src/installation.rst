Installation
============

We provide the package via PyPI. To install ``mitransient`` you need to run:

.. code-block:: bash

      pip install mitransient

If you have installed Mitsuba 3 via ``pip`` you will only have access to the ``llvm_ad_rgb`` and ``cuda_ad_rgb`` variants. If you want to use other variants (e.g. NLOS simulations can greatly benefit from the ``llvm_mono`` variant which only propagates one wavelength), then we recommend that you compile Mitsuba 3 yourself `following this tutorial <https://mitsuba.readthedocs.io/en/latest/src/developer_guide/compiling.html>`_ and enable the following variants: ``["scalar_mono", "llvm_mono", "llvm_ad_mono", "cuda_mono", "cuda_ad_mono", "scalar_rgb", "llvm_rgb", "llvm_ad_rgb", "cuda_rgb", "cuda_ad_rgb"]``.

Requirements
------------

* ``Python >= 3.8``
* ``Mitsuba3 >= 3.5.0``
* (optional) For computation on the GPU: ``Nvidia driver >= 495.89``
* (optional) For vectorized / parallel computation on the CPU: ``LLVM >= 11.1``

After installation
------------------

At this point, you should be able to ``import mitsuba`` and ``import mitransient`` in your Python code (careful about setting the correct ``PATH`` environment variable if you have compiled Mitsuba 3 yourself, see the section below).

For NLOS data capture simulations, see `https://github.com/diegoroyo/tal <https://github.com/diegoroyo/tal>`_. ``tal`` is a toolkit that allows you to create and simulate NLOS scenes with an easier shell interface instead of directly from Python.

If you use your own Mitsuba 3
-----------------------------

If you have opted for using a custom (non-default installation through ``pip``) Mitsuba 3, you have several options for it. The idea here is to be able to control which version of Mitsuba will be loaded on demand.

* One solution is to directly execute ``setpath.sh`` provided after the compilation of the Mitsuba 3 repo `(More info) <https://mitsuba.readthedocs.io/en/latest/src/developer_guide/compiling.html>`_. This shell script will modify the ``PATH`` and ``PYTHONPATH`` variables to load first this version of Mitsuba.
* Another solution following the previous one is to directly set yourself the ``PYTHONPATH`` environment variable as you wish.
* Another solution for having a custom version globally available is by using ``pip install . --editable``. This will create a symlink copy of the package files inside the corresponding ``site-packages`` folder and will be listed as a package installed of ``pip`` and will be available as other packages installed. If you recompile them, you will still have the newest version directly to use. Please follow these instructions:
  * Go to ``<mitsuba-path>/mitsuba3/build/python/drjit`` and execute ``pip install . --editable``.
  * Go to ``<mitsuba-path>/mitsuba3/build/python/mitsuba`` and execute ``pip install . --editable``.
* If you are a user of Jupyter Notebooks, the easiest solution will be to add the following snippet of code to modify the notebook's ``PYTHONPATH``:

.. code-block:: python

    import sys
    sys.path.insert(0, '<mitsuba-path>/mitsuba3/build/python')
    import mitsuba as mi

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# Setup mitsuba and add mitransient to path


import sys
import os

sys.path.insert(0, os.path.abspath('..'))

import mitsuba as mi
import drjit as dr
mi.set_variant('llvm_ad_rgb')
import mitransient as mitr

# Add for custom plugins / directives
sys.path.append(os.path.abspath('exts'))

rst_prolog = r"""
.. role:: paramtype

.. role:: monosp

.. |spectrum| replace:: :paramtype:`spectrum`
.. |texture| replace:: :paramtype:`texture`
.. |float| replace:: :paramtype:`float`
.. |bool| replace:: :paramtype:`boolean`
.. |int| replace:: :paramtype:`integer`
.. |false| replace:: :monosp:`false`
.. |true| replace:: :monosp:`true`
.. |string| replace:: :paramtype:`string`
.. |bsdf| replace:: :paramtype:`bsdf`
.. |phase| replace:: :paramtype:`phase`
.. |point| replace:: :paramtype:`point`
.. |vector| replace:: :paramtype:`vector`
.. |transform| replace:: :paramtype:`transform`
.. |volume| replace:: :paramtype:`volume`
.. |tensor| replace:: :paramtype:`tensor`

.. |drjit| replace:: :monosp:`drjit`
.. |numpy| replace:: :monosp:`numpy`

.. |nbsp| unicode:: 0xA0
   :trim:

.. |exposed| replace:: :abbr:`P (This parameters will be exposed as a scene parameter)`
.. |differentiable| replace:: :abbr:`∂ (This parameter is differentiable)`
.. |discontinuous| replace:: :abbr:`D (This parameter might introduce discontinuities. Therefore it requires special handling during differentiation to prevent bias))`

"""

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'mitransient'
copyright = '2024, Diego Royo, Miguel Crespo, Jorge Garcia-Pueyo'
author = 'Diego Royo, Miguel Crespo, Jorge Garcia-Pueyo'
release = mitr.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []
extensions.append('sphinx.ext.autodoc')
extensions.append('sphinx.ext.coverage')
extensions.append('sphinx.ext.napoleon')
extensions.append('sphinx.ext.intersphinx')
extensions.append('sphinxcontrib.youtube')
extensions.append('sphinx_copybutton')
extensions.append('nbsphinx')
nbsphinx_execute = 'never'
extensions.append('sphinx_gallery.load_style')
extensions.append('pluginparameters')


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'src/plugin_reference/section_*.rst']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_title = 'mitransient'
html_theme = 'furo'
html_static_path = ['_static']

# Force pygments style in dark mode back to the light variant
pygments_dark_style = 'tango'

# If true, links to the reST sources will be added to the sidebar.
html_show_sourcelink = False

html_theme_options = {
   # Disable edit button on read the docs
   "top_of_page_button": None,
}

# Generate the documentation from the source files
from docs import generate_plugin_doc

generate_plugin_doc.generate(
    'src/plugin_reference', 'generated/plugin_reference')

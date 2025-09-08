Differentiable transient rendering
==================================

We extend the capabilities of Mitsuba to support differentiable rendering in the time domain.
This allows you to compute gradients of transient renders with respect to scene parameters,
enabling applications such as inverse rendering and optimization in the transient domain.

.. nbgallery::
      :maxdepth: 1
      :caption: Differntiable transient rendering tutorials
      :hidden:
      
      ../examples/diff-transient/backward_transient
      ../examples/diff-transient/backward_steady
      ../examples/diff-transient/forward_inverse_rendering_cbox
      ../examples/diff-transient/forward_inverse_rendering_staircase


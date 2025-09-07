Time-resolved polarization
==========================

Tracking the polarization of light in transient rendering is posible using the ``*_polarized`` variants of Mitsuba. ``mitransient`` is configured to support polarized variants without requiring any additional changes from the user. Additionally, we provide a suite of visualization functions that help you visualize the polarization in your transient renders.

We also provide support for transient polarized NLOS simulation [PueyoCiutad2024]_.

.. nbgallery::
      :maxdepth: 1
      :caption: Polarization tutorials
      :hidden:
      
      ../examples/polarization/render_cbox_polarized_and_visualization
      ../examples/polarization/render_cbox_polarized_and_visualization_steady
      ../examples/polarization/transient_nlos_polarization


.. [PueyoCiutad2024] Pueyo-Ciutad, O., Marco, J., Schertzer, S., Christnacher, F., Laurenzis, M., Gutierrez, D., & Redo-Sanchez, A. (2024, December). Time-Gated Polarization for Active Non-Line-Of-Sight Imaging. In SIGGRAPH Asia 2024 Conference Papers (pp. 1-11).


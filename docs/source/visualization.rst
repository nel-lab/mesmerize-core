.. _visualization:

Visualization
*************

A number of visualization are available. For anything which doesn't require fast user feedback or static plots you can probably just use ``matplotlib`` and ``seaborn``. For visualization of movies, such as after motion correction, the reconstructed movie after CNMF, residuals etc, or drawing thousands of ROIs, we highly recommend ``fastplotlib`` and ``mesmerize-viz``. The demo notebooks provide extensive examples using ``fastplotlib`` to visualize the motion corrected movie, spatial contours, reconstructed movie, residuals, and reconstructed background.

| fastplotlib: https://github.com/kushalkolar/fastplotlib
| mesmerize-viz: https://github.com/kushalkolar/mesmerize-viz
| demo notebooks: https://github.com/nel-lab/mesmerize-core/tree/master/notebooks

A napari plugin exists, ``mesmerize-napari``, but it is not as fast and versatile as ``fastplotlib``: https://github.com/nel-lab/mesmerize-napari

Intro
-----

The ``kszx`` code is split across two github repos:

  - "Core" repo containing source code for ``kszx`` package:

    https://github.com/kmsmith137/kszx.

  - Auxiliary repo containing jupyter notebooks:

    https://github.com/kmsmith137/kszx_notebooks.

  - (This two-repo split is because jupyter notebooks tend to result in large github repos,
    and I wanted to keep the "core" kszx repo small.)

The goal of the ``kszx`` package is to make a unified framework for different kSZ pipelines/projects
which use a "3-d Cartesian" approach.

 - Define low-level "building blocks" with simple interfaces, for example :ref:`fft wrappers`
   or :ref:`gridding/interpolation kernels <gridding/interpolation>`.

   Status: this currently exists in "rough draft" form, but could probably use improvement.
   Feel free to suggest (or make) changes!
   
 - Make low-level building blocks run very fast.
   Computational cost is emerging as an issue -- I think we can make the building blocks very fast,
   with some strategically written C++ kernels.

   Status: haven't really started yet, but I think this can be done quickly.
   
 - Define high-level functions that can be combined to make pipelines.

   Status: haven't really started yet -- currently these high-level functions are scattered
   between different jupyter notebooks, and haven't been incoporated into the ``kszx`` package.

Authors
-------

 - Selim Hotinli (https://github.com/selimhotinli)
 - Kendrick Smith (https://github.com/kmsmith137/)
 - Your name here??

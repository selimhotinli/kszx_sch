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

Data files
----------

The kszx library contains functions which read data files from disk.
For example, :func:`kszx.sdss.read_galaxies()` reads an SDSS galaxy catalog.
Such functions generally contain a ``download`` argument (False by default) which
will auto-download the data file, if it is not found on disk.

All auto-downloaded files are saved in a directory tree whose root is given by:

  - The environment variable ``$KSZX_DATA_DIR``, if defined.

  - Otherwise, ``$HOME/kszx_data``.

Therefore, here are some options for setting up your auto-download directory:

  - If you want to store data in ``$HOME/kszx_data``, do ``mkdir -p $HOME/kszx_data``.
    
  - If you want to use a different directory, do ``ln -s /some/other/dir $HOME/kszx_data``.
    
    (For example, on Kendrick's desktop, ``/data`` already contains a lot of auto-downloaded
    kszx data files, so you can do ``ln -s /data $HOME/kszx_data``.)
    
  - If you want to use a different directory, do ``ln -s /some/other/dir $HOME/kszx_data``. 

The reason I'm discussing this in so much detail is that auto-downloaded files can really
take up a lot of space!

Not all parts of the kszx library need to use the auto-download directory (just functions
which read large data files), so you don't need to set it up right away.

Authors
-------

 - Selim Hotinli (https://github.com/selimhotinli)
 - Yurii Kvasiuk (https://github.com/ykvasiuk)
 - Edmond Chaussidon (https://github.com/echaussidon)
 - Alex Laguë (https://github.com/alexlague)
 - Mat Madhavacheril (https://github.com/msyriac)
 - Kendrick Smith (https://github.com/kmsmith137)
 - Your name here??

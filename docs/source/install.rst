Installation
------------

- Currently, kszx works on Linux but not Apple!
  If you need Apple support, let me know and I can prioritize.
  
- Step 1. (Optional but recommended.) Install some prerequisites with conda.

  This may not be necessary, but maximizes the probability that the ``pip install`` in step 2
  will work. (If you skip step 1, then ``pip install`` will install a long chain of dependencies,
  some of which have hidden dependence on tools installed on your system, e.g. compilers, automake,
  libtool.)

  You can either install the "precomputed" conda env in ``kszx/environment.yml``::

     git clone https://github.com/kmsmith137/kszx
     cd kszx
     conda env create -n kszx -f environment.yml
     conda activate kszx

  Or build your own conda env with::

     # Note: gcc 14 conflicts with camb (or at least conda thinks so), so force gcc==13.
     # Note: jupyterlab is only needed if you're running jupyterlab locally.
     # Note: sphinx packages are only needed if you're building sphinx docs locally.

     conda create -n kszx -c conda-forge \
       gxx_linux-64==13.3.0 \
       python==3.12.6 pybind11 \
       jupyterlab \
       sphinx sphinx-math-dollar \
       h5py python-wget astropy camb fitsio healpy pixell

    conda activate kszx

  The second approach takes longer, but may be preferred if you want to customize/extend the
  environment.

  (Note: if you don't have conda installed, I recommend the ``miniforge`` version of conda,
  available at https://github.com/conda-forge/miniforge, table heading "Miniforge3".)

- Step 2: Install kszx with pip.
  Do one of the following:

  - My preferred 'pip install' is an "editable install" (pip install -e), which is
    nice because you can edit source files without doing 'pip install' again::

      # Editable install
      git clone https://github.com/kmsmith137/kszx
      cd kszx
      pip install -v -e .

  - For a non-editable install, omit the ``-e`` flag::
    
      # Non-editable install
      git clone https://github.com/kmsmith137/kszx
      cd kszx
      pip install -v .

  - Finally, there is a kszx version on PyPI (https://pypi.org/project/kszx/),
    so you can install without cloning the github repo, but it may not be the
    most current::

      # PyPI install (may not be current)
      pip install -v kszx

- Step 3: I recommend testing that the install worked, with::

    # Just runs some unit tests
    python -m kszx test

- Step 4 (optional): install Mat Madhavacheril's ``hmvec`` library. This library
  isn't currently used in ``kszx`` itself, but is used in some of the notebooks
  (https://github.com/kmsmith137/kszx_notebooks/)::

    # Optional: install hmvec
    pip install -v git+https://github.com/simonsobs/hmvec
 
- Step 5: You'll probably want to use kszx from a jupyter notebook, so you'll
  want to make sure you can ``import kszx`` from jupyter.

  The details of the setup will depend on how you're running jupyter, so
  it's hard for me to give general instructions, but if you need help let me know.
  If you're running ``jupyterlab`` from the command line yourself, then a
  straightforward approach is to simply run the ``jupyterlab`` executable from
  an environment that has ``kszx`` installed. (Note that the conda env in step 1
  contains jupyterlab already, but if you skipped step 1, then you'll need to do
  ``pip install jupyterlab``.)
  
 
  
  
  

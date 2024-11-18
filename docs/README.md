### Readthedocs conda environment:

```
conda create -n kszx_rtd -c conda-forge \
   python==3.12 \
   pybind11 automake gxx_linux-64 \
   sphinx sphinx-math-dollar \
   h5py python-wget astropy camb fitsio healpy

conda activate kszx_rtd
conda env export -f environment_rtd.yml

# edit environment_rtd.yml, and add pixell (via pip) by hand:
  - pip:
    - pixell >= 0.26.1
    
# Note that pymangle is not required to build documentation.
```

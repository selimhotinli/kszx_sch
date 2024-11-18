### Note to myself on readthedocs conda environment:

```
conda create -n kszx_rtd -c conda-forge \
   python==3.12.6 \
   gxx_linux-64 pybind11 \
   sphinx sphinx-math-dollar \
   h5py python-wget astropy camb fitsio healpy pixell

conda activate kszx_rtd
conda env export -f environment_rtd.yml
```

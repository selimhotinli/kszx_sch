conda create -n kszx -c conda-forge \
   python jupyterlab jupytext pybind11 automake gxx_linux-64 \
   h5py python-wget astropy camb fitsio healpy \
   bioconda::snakemake

conda activate ksz

pip install pixell

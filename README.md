### Environment:

```
conda create -n kszx -c conda-forge \
   python==3.12.6 \
   jupyterlab jupytext \
   gxx_linux-64 pybind11 \
   sphinx sphinx-math-dollar \
   h5py python-wget astropy camb fitsio healpy pixell
```

### Jupyterlab:

Doing this the "wrong" way for now -- will set up jupyterhub later.

One-time initializations:
```
# On server, starts interactive dialog to set password
jupyter lab password 

# In server bashrc
alias iamtunnel='echo && echo && echo "      THIS IS AN SSH TUNNEL -- DO NOT CLOSE THIS WINDOW     " && echo && echo && sleep 10000000'

# In laptop bashrc
tunnel_mango='ssh -L 8888:localhost:8888 mango'
```

On server:
```
tmux new -s jupyter   # subsequent commands are in this tmux window
conda activate kszx   # since jupyter is not in base conda env
jupyter lab list      # check whether jupyter is already running
jupyter lab --no-browser --port=8888 --notebook-dir=/home/kmsmith/git/kszx
```

On laptop (to connect):
```
tunnel_mango
iamtunnel
```

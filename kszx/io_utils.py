import os
import h5py
import fitsio
import pickle   # FIXME cPickle?
import functools
import numpy as np

# Avoid name collision with io_utils.wget() below.
import wget as wget_module


def read_pickle(filename):
    print(f'Reading {filename}')
    with open(filename, 'rb') as f:
        return pickle.load(f)


def write_pickle(filename, x):
    mkdir_containing(filename)
    print(f'Writing {filename}')    
    with open(filename, 'wb') as f:
        pickle.dump(x, f)


def read_npy(filename, verbose=True):
    print(f'Reading {filename}')
    return np.load(filename, allow_pickle=True)


def write_npy(filename, arr, verbose=True):
    mkdir_containing(filename)
    print(f'Writing {filename}')    
    np.save(filename, arr)

    
def read_npz(filename, verbose=True):
    """
    Reminder: returns a dictionary-like object (NpzFile) which maps names to arrays.

    Some arrays may have special names 'arr_0', 'arr_1', ... 
    This happens for arrays which were positional (not keyword) arguments to write_npz().
    """

    # read_npy() and read_npz() can be the same function,
    # since np.load() works for both .npy and .npz files.
    return read_npy(filename, verbose=verbose)


def write_npz(filename, *args, **kwds):
    """
    Example: write_npz('file.npz', a, b, xarr=x, yarr=y), where (a,b,x,y) are numpy arrays.

    In this example, when 'file.npz' is read (with np.load() or kszx.read_npz()), the return 
    value is a dictionary-like object (NpzFile) with keys 'arr_0', 'arr_1', 'xarr', 'yarr'.
    Note that the positional arguments (a,b) get special names 'arr_0', 'arr_1'.
    """

    mkdir_containing(filename)
    print(f'Writing {filename}')    
    np.savez(filename, *args, **kwds)


####################################################################################################


@functools.cache
def get_data_dir(name):
    """
    Usage: get_data_dir('desils') returns

      - the value of the environment variable DESILS_DATA_DIR if defined,

      - otherwise $(root)/deslis, where $(root) is:
          - the value of the environment variable KSZX_ROOT_DATA_DIR if defined,
          - otherwise, /data

    Special case: get_data_dir('kszx_root') returns $(root) as defined above. 
    (Used internally, when calling get_data_dir() recursively.)
    """

    env_varname = f'{name.upper()}_DATA_DIR'
    
    if env_varname in os.environ:
        ret = os.environ[env_varname]
        print(f"Using {name} data directory {ret} from environment variable {env_varname}")
        return ret

    if name.upper() == 'KSZX_ROOT':
        ret = '/data'
    else:
        ret = os.path.join(get_data_dir('kszx_root'), name)
        
    print(f"Using default {name} data directory '{ret}' (set environment variable {env_varname} to override)")
    return ret


def mkdir_containing(filename):
    assert isinstance(filename, str)
    
    d = os.path.dirname(filename)
    if (d == '') or os.path.exists(d):
        return
    
    print(f'Creating directory {d}')
    os.makedirs(d, exist_ok=True)
    

def wget(filename, url):
    if os.path.exists(filename):
        print(f'File {filename} already exists -- skipping download')
        return
    
    mkdir_containing(filename)
    print(f'Downloading {url} -> {filename}')
    
    wget_module.download(url, out=filename)
    print()   # extra newline after progress bar
    assert os.path.exists(filename)


####################################################################################################


def show_file(filename, file_type=None, file=None):
    """Prints (to stdout) a simple human-readable summary of file with given name.

    The file_type can be one of:

      pkl        python pickle
      npy        numpy array
      npz        numpy NpzFile
      h5, hdf5   HDF5 file
      fits       FITS file

    If file_type is None, then the file_type is determined from the filename in 
    a brain-dead way (e.g. if filename='x.pkl', then file_type='pkl').

    Can be invoked from command line with 'python -m kszx show <filename>'.
    """

    assert isinstance(filename, str)
        
    if file_type is None:
        b = os.path.basename(filename)
        i = b.rfind('.')
        if (i < 0) or (i >= len(b)-1):
            raise RuntimeError(f"Couldn't determine file_type from {filename=}")
        file_type = b[(i+1):]

    if file_type == 'pkl':
        x = read_pickle(filename)
        print(x, file=file)

    elif file_type == 'npy':
        x = read_npy(filename)
        print('Shape:', x.shape)
        print(x, file=file)

    elif file_type == 'npz':
        _show_npz(filename, file=file)

    elif file_type in [ 'h5', 'hdf5' ]:
        _show_h5(filename, file=file)

    elif file_type in [ 'fits' ]:
        _show_fits(filename, file=file)
            
    else:
        raise RuntimeError(f"kszx.show_file(): unsupported file_type {file_type}")
        

def _show_npz(x, file=None, indent='  '):
    """Intended to be called through show_file(), or via command line.

    Prints (to stdout) a simple human-readable summary of 'x', which can be
    either a filename or a NpzFile (class numpy.lib.npyio.NpzFile).
    """

    if isinstance(x, str):
        x = read_npz(x)

    if not isinstance(x, np.lib.npyio.NpzFile):
        raise RuntimeError('kszx._show_npz(): argument must be either string or NpzFile')

    print(x, file=file)
    
    for k,v in x.items():
        print(f'{indent}{k}: shape={v.shape}, min={np.min(v)}, max={np.max(v)}', file=file)


def _show_h5_attrs(x, prefix, file):
    for k,v in sorted(x.attrs.items()):
        print(f"{prefix}Attribute '{k}': {v}", file=file)
    

def _show_h5(x, prefix=None, file=None, indent='  '):
    """Intended to be called through show_file(), or via command line.

    Prints (to stdout) a simple human-readable summary of 'x', which can be
    either a filename, an h5py.File, or an h5py.Group.

    The default value of the 'indent' argument indents by two spaces at every level.
    """

    if prefix is None:
        prefix = ''
    if file is None:
        file = sys.stdout
    
    if isinstance(x, str):
        with h5py.File(x) as f:
            print(f"{prefix}File '{x}'", file=file)
            _show_h5(f, prefix + indent, file, indent)
        return

    _show_h5_attrs(x, prefix, file)

    # x.items() = "all groups and datasets in a single dict".
    # First pass: loop over datasets.
    
    for k,v in sorted(x.items()):
        if isinstance(v, h5py.Dataset):
            try:
                s = f', min={np.min(v)}, max={np.max(v)}'
            except:
                s = ''
            print(f"{prefix}Dataset '{k}': shape={v.shape}, dtype={v.dtype}{s}", file=file)
            _show_h5_attrs(v, prefix + indent, file, indent)
        elif not isinstance(v, h5py.Group):
            print(f"{prefix}Unrecognized element '{k}': type={type(v)}", file=file)

    # Second pass: loop over groups

    for k,v in sorted(x.items()):
        if isinstance(v, h5py.Group):
            print(f"{prefix}Group '{k}'", file=file)
            _show_h5(v, prefix + indent, file, indent)


def _show_fits(x, file=None):
    """Intended to be called through show_file(), or via command line.

    Prints (to stdout) a simple human-readable summary of 'x', which can be
    either a filename, or a fitsio.FITS object.

    This quick-and-dirty implementation just calls __str__() functions in the
    fitsio python module (which is pretty good).
    """

    if isinstance(x, str):
        with fitsio.FITS(x) as f:
            _show_fits(f)

    elif isinstance(x, fitsio.FITS):
        for hdu in x:
            print(hdu, file=file)

    else:
        raise RuntimeError('kszx._show_fits(): argument must be a string, or of type fitsio.FITS')

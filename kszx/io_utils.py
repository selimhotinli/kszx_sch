import os
import functools

# Avoid name collision with io_utils.wget() below.
import wget as wget_module


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

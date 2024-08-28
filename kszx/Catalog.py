import h5py
import copy
import healpy
import numpy as np

from . import io_utils    # mkdir_containing()


class Catalog:
    """
    A Catalog contains the following members:

       self.size        (integer)
       self.col_names   (list of strings)
       self.name        (string or None)
       self.filename    (string or None)

    Additionally, for each column name (in self.col_names), the Catalog
    contains a member with the corresponding name, whose value is a 1-d array
    of length self.size.

    Some standard column names:
       ra_deg      right ascension (degrees)
       dec_deg     declination (degrees)
       z           redshift
       zerr        redshift error (if photometric)
       rmag        r-band magnitude (and similarly for g-band, z-band, etc.)
    """
    
    def __init__(self, cols=None, name=None, filename=None, size=0):
        """
        The 'cols' argument should be a dictionary col_name -> col_data.
        The 'name' argument is an optional string.
        """

        self.size = size
        self.name = name
        self.col_names = []
        self.filename = filename

        if cols is not None:
            for (col_name, col_data) in cols.items():
                self.add_column(col_name, col_data)

    
    def add_column(self, col_name, col_data):        
        assert isinstance(col_name, str)
        assert len(col_name) > 0
        assert not col_name.startswith('_')
        assert col_name not in [ 'size', 'name', 'filename', 'col_names' ]
        assert col_name not in self.col_names

        col_data = np.asarray(col_data)
        assert col_data.ndim == 1

        if (len(self.col_names) > 0) or (self.size != 0):
            assert len(col_data) == self.size
        else:
            self.size = len(col_data)
        
        self.col_names.append(col_name)
        setattr(self, col_name, col_data)


    def remove_column(self, col_name):
        assert isinstance(col_name, str)
        assert len(col_name) > 0
        assert not col_name.startswith('_')
        assert col_name not in [ 'size', 'name', 'filename', 'col_names' ]
        assert col_name in self.col_names

        self.col_names.remove(col_name)
        delattr(self, col_name)


    def apply_boolean_mask(self, mask, name=None, inplace=True):
        if not inplace:
            self = self.shallow_copy()
                
        mask = np.asarray(mask)
        assert mask.dtype == bool
        if mask.shape != (self.size,):
            raise RuntimeError(f'Catalog.apply_boolean_mask(): expected {mask.shape=} to equal {(self.size,)=}')

        new_size = np.sum(mask)

        if name is not None:
            print(f'{name}: {self.size} -> {new_size}')
        
        for col_name in self.col_names:
            col = getattr(self, col_name)
            setattr(self, col_name, col[mask])
            assert getattr(self,col_name).shape == (new_size,)

        self.size = new_size
        return self


    def apply_redshift_cut(self, zmin, zmax):
        assert 'z' in self.col_names
        
        if zmin is not None:
            self.apply_boolean_mask(self.z >= zmin, name=f'z >= {zmin}')
        if zmax is not None:
            self.apply_boolean_mask(self.z <= zmax, name=f'z <= {zmax}')


    def to_healpix(self, nside, weights=None):
        """Returns 1-d float64 array with length 12*nside^2."""
        
        npix = healpy.nside2npix(nside)
        ipix = healpy.ang2pix(nside, self.ra_deg, self.dec_deg, lonlat=True)
        ret = np.zeros(npix)

        if weights is not None:
            weights = np.asarray(weights)
            assert weights.shape == (self.size,)
            
        # FIXME slow non-vectorized loop, write C++ helper function?
        # (I don't think there is a numpy function that helps.)

        if weights is None:
            for p in ipix:
                ret[p] += 1.0
        else:
            for i in range(self.size):
                ret[ipix[i]] += weights[i]
                
        return ret


    def generate_batches(self, batchsize, verbose=True):
        """
        Splits catalog into subcatalogs no larger than 'batchsize'.
        If 'batchsize' is None, the catalog will be "split" into a single subcatalog.

        If 'verbose' is True, and the catalog is split into more than one batch,
        then a progress indicator will be shown.
        """

        nbatches = 1

        if batchsize is not None:
            assert batchsize > 0
            nbatches = (self.size + batchsize - 1) // batchsize

        if nbatches == 1:
            yield self
            return
        
        for b in range(nbatches):
            s = f'subcatalog {b}/{nbatches}'
            if verbose:
                print('    ', s)  # progress indicator
            if self.name is not None:
                s = f'{self.name} [{s}]'

            i = (b * self.size) // nbatches
            j = ((b+1) * self.size) // nbatches
            assert i < j

            # Subcatalogs are constructed by slicing, not "deep copy".
            yield Catalog(
                cols = { k: getattr(self,k)[i:j] for k in self.col_names },
                name = s,
                size = j-i
            )

            
    def shallow_copy(self):
        return Catalog(
            cols = { s: getattr(self,s) for s in self.col_names },
            name = self.name,
            filename = self.filename,
            size = self.size
        )


    @staticmethod
    def concatenate(catalog_list, name=None, destructive=False):
        assert len(catalog_list) > 0
        assert all(isinstance(x, Catalog) for x in catalog_list)

        if any(set(g.col_names) != set(catalog_list[0].col_names) for g in catalog_list):
            raise RuntimeError('Catalog.concatenate(): all Catalogs must have the same column names')

        # copy.copy() is necessary if destructive=True
        col_names = copy.copy(catalog_list[0].col_names)
        cols = { }

        for k in col_names:
            cols[k] = np.concatenate([getattr(g,k) for g in catalog_list])
            if destructive:
                for g in catalog_list:
                    g.remove_column(k)
            
        return Catalog(cols=cols, name=name)
        
    
    @staticmethod
    def from_h5(filename):
        print(f'Reading {filename}')
        
        with h5py.File(filename, 'r') as f:
            catalog = Catalog(
                cols = { k: np.asarray(v) for k,v in f.items() },
                name = f.attrs.get('name', default=None),
                filename = filename,
                size = f.attrs['size']
            )

        catalog._announce_file_read()
        return catalog


    def write_h5(self, filename):
        io_utils.mkdir_containing(filename)
        print(f'Writing {filename}')
        
        with h5py.File(filename, 'w') as f:
            f.attrs.create('size', self.size)
            
            if self.name is not None:
                f.attrs.create('name', self.name)
                
            for col_name in self.col_names:
                col = getattr(self, col_name)
                assert col.shape == (self.size,)
                f.create_dataset(col_name, data=col)

        self._announce_file_write(filename)


    def write_txt(self, filename):
        """Write catalog in an ad hoc text format, intended for human readability."""

        io_utils.mkdir_containing(filename)
        print(f'Writing {filename}')
            
        cols = [ getattr(self,col_name) for col_name in self.col_names ]
        ncols = len(cols)
                         
        with open(filename,'w') as f:
            if self.name is not None:
                print(f'# Catalog name: {self.name}', file=f)
            
            print(f'# Columns are as follows:', file=f)
            for i, col_name in enumerate(col_names):
                print(f'# Column {i}: {col_name}')

            for i in range(self.size):
                for j in range(ncols):
                    print('  ', cols[j][i], end=None, file=f)
                print(file=f)

        self._announce_file_write(filename)
            

    def _announce_file_read(self):
        s1 = '' if (self.name is None) else f'{self.name}: '
        s2 = '' if (self.filename is None) else f' from {self.filename}'
        print(f'{s1}Read {self.size} galaxies{s2}, columns {self.col_names}')

    def _announce_file_write(self, filename):
        s = '' if (self.name is None) else f'{self.name}: '
        print(f'{s}Wrote {self.size} galaxies to {filename}, columns {self.col_names}')
    
    def __str__(self):
        s1 = '' if (self.name is None) else f'name={self.name}, '
        s2 = '' if (self.filename is None) else f', filename={self.filename}'
        return f'Catalog({s1}size={self.size}, col_names={self.col_names}{s2})'

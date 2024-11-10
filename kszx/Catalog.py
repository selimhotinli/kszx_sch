import h5py
import copy
import fitsio
import numpy as np
import pixell.enmap

from . import utils
from . import io_utils


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

    Note: here are some functions which make maps from catalogs:

       kszx.healpix_utils.map_from_catalog()   # catalog -> 2d healpix map
       kszx.pixell_utils.map_from_catalog()    # catalog -> 2d pixell maps
       kszx.grid_points()                      # catalog -> 3d map
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


    def apply_boolean_mask(self, mask, name=None, in_place=True):
        if not in_place:
            self = self.shallow_copy()
                
        mask = np.asarray(mask)
        assert mask.dtype == bool
        if mask.shape != (self.size,):
            raise RuntimeError(f'Catalog.apply_boolean_mask(): expected {mask.shape=} to equal {(self.size,)=}')

        new_size = np.sum(mask)

        if name is not None:
            p = 100 * new_size / self.size
            print(f'{name}: {self.size} -> {new_size} [{p:.05}% retained]')
        
        for col_name in self.col_names:
            col = getattr(self, col_name)
            setattr(self, col_name, col[mask])
            assert getattr(self,col_name).shape == (new_size,)

        self.size = new_size
        return self


    def apply_redshift_cut(self, zmin, zmax, in_place=True):
        assert 'z' in self.col_names

        mask1 = (self.z >= zmin) if (zmin is not None) else np.ones(self.size)
        mask2 = (self.z <= zmax) if (zmax is not None) else np.ones(self.size)
        
        mask = np.logical_and(mask1, mask2)
        name = f'Redshift cut: {zmin=} {zmax=}'
        return self.apply_boolean_mask(mask, name=name, in_place=in_place)


    def get_xyz(self, cosmo):
        """Returns shape (N,3) array. The 'cosmo' arg should be an instance of kszx.Cosmology.

        Can be used as 'points' array in kszx.interpolate_points(), kszx.grid_points(), or
        kszx.BoundingBox constructor.
        
        Implementation: just a wrapper around:

           kszx.Cosmology.chi()         computes radial distance chi from 'z' column of catalog
           kszx.utils.ra_dec_to_xyz()   computes xyz coords from chi + ('ra_deg','dec_deg') cols
        """

        assert 'ra_deg' in self.col_names
        assert 'dec_deg' in self.col_names
        assert 'z' in self.col_names
        
        chi = cosmo.chi(z=self.z)
        xyz = utils.ra_dec_to_xyz(self.ra_deg, self.dec_deg, r=chi)
        return xyz
    

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

        
    @staticmethod
    def from_fits(filename, col_name_pairs, name=None):
        """Returns a Catalog object. Used to read SDSS, DESILS catalogs.
        
        The 'col_name_pairs' arg should be a list of pairs (col_name, fits_col_name).
        See sdss.py and desils_lry.py for examples."""

        if name is None:
            name = filename
            
        print(f'Reading {filename}')
        catalog = Catalog(name=name, filename=filename)
        
        with fitsio.FITS(filename) as f:
            for (col_name, fits_col_name) in col_name_pairs:
                catalog.add_column(col_name, f[1].read(fits_col_name))
                
        catalog._announce_file_read()
        return catalog

    
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

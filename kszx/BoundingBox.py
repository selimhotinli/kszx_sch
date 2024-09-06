import io   # StringIO
import numpy as np

from . import utils   # ra_dec_to_xyz()

from .Box import Box
from .Catalog import Catalog
from .Cosmology import Cosmology


class BoundingBox(Box):
    def __init__(self, catalog, cosmo, pixsize, rpad):
        """
        Assumes catalog has 'ra_deg', 'dec_deg', and 'z' members defined.
        Suggest something like 200 Mpc for 'rpad'.

        Inherits the following members from Box: XXX
        """
        
        assert isinstance(catalog, Catalog)
        assert isinstance(cosmo, Cosmology)
        assert catalog.size > 0
        assert pixsize > 0
        assert rpad >= 0

        assert np.all(catalog.z > 0.)
        self.zmin = np.min(catalog.z)
        self.zmax = np.max(catalog.z)
        self.chi_min = cosmo.chi(z = self.zmin)
        self.chi_max = cosmo.chi(z = self.zmax)

        nxyz = utils.ra_dec_to_xyz(catalog.ra_deg, catalog.dec_deg)  # shape (ngal,3)
        nxyz_min = np.min(nxyz, axis=0)  # shape (3,)
        nxyz_max = np.max(nxyz, axis=0)  # shape (3,)
        del nxyz

        # Shape (4, 3)
        xyz_corners = np.array([
            self.chi_min * nxyz_min,
            self.chi_min * nxyz_max,
            self.chi_max * nxyz_min,
            self.chi_max * nxyz_max
        ])

        # These arrays are shape (3,)
        xyz_min = np.min(xyz_corners, axis=0)  # shape (3,)
        xyz_max = np.max(xyz_corners, axis=0)  # shape (3,)
        cpos = (xyz_min + xyz_max) / 2.
        bs = (xyz_max - xyz_min)    # boxsize

        # These arrays are shape (3,)
        self.npix_prepad = np.array(np.ceil(bs / pixsize), dtype=int)
        self.npix_preround = np.array(np.ceil((bs+rpad) / pixsize), dtype=int)
        npix = 4 * self.npix_preround  # sentinel
        
        for p in [ 1,3,5,9,15,27 ]:
            q = (self.npix_preround - 0.5) / p
            r = np.ceil(np.log2(q))
            r = np.array(r, dtype=int)
            npix = np.minimum(npix, p * 2**r)

        Box.__init__(self, npix, pixsize, cpos=cpos)

        
    def __str__(self):
        with io.StringIO() as f:
            print(f'BoundingBox(', file=f)
            self._print_box_members(f, end='')
            print(f'    zmin = {self.zmin:.04f}, zmax = {self.zmax:.04f},', file=f)
            print(f'    chi_min = {self.chi_min:.02f}, chi_max = {self.chi_max:.02f},', file=f)
            print(f'    npix_prepad = {self.npix_prepad},', file=f)
            print(f'    npix_preround = {self.npix_preround}', file=f)
            print(f')', file=f)
            return f.getvalue()

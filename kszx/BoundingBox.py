import io   # StringIO
import numpy as np

from . import utils

from .Box import Box


class BoundingBox(Box):
    def __init__(self, points, pixsize, rpad):
        """Subclass of Box which contains a specified set of points + padding.

        Constructor args:

            points: numpy array with shape (npoints, ndim).
              Contains galaxy locations, in coordinate system with observer at origin.
              Usually obtained by calling Catalog.get_xyz(cosmo) -- see example below.

            pixsize (float): pixel length in same units as 'points'.

            rpad (float): box padding in same units as 'points' (suggest ~200 Mpc).

        Inherits the following members from Box: 

           ndim           number of dimensions N
           npix           real-space map shape, represented as length-N array
           pixsize        pixel side length in user-defined coordinates (scalar, i.e. pixels must be square)
           boxsize        box side lengths in user-defined coordinates (equal to npix * pixsize)
           nk             Fourier-space map shape, represented as length-N array
           kfund          lowest frequency on each axis (length-N array, equal to 2pi/boxsize)
           knyq           Nyquist frequency (scalar, equal to pi/pixsize)
           box_volume     scalar box volume (equal to prod(boxsize))
           pixel_volume   scalar pixel volume (equal to pixsize^N)
           cpos           location of box center in observer coordinates (length-N array)
           lpos           location of lower left corner (equal to cpos- (npix-1)*pixsize/2)
           rpos           location of upper right corner (equal to cpos + (npix-1)*pixsize/2)
           real_space_shape     same as 'npix', but represented as tuple instead of 1-d array
           fourier_space_shape  same as 'nk', but represented as tuple instead of 1-d array

        Example: make bounding box which contains galaxies + randoms.
        
            # Setup: 'gcat' and 'rcat' are objects of type Catalog.
            # 'cosmo' is an object of type Cosmology.

            gcat_xyz = gcat.get_xyz(cosmo)    # shape (ngal,3)
            rcat_xyz = rcat.get_xyz(cosmo)    # shape (rand,3)

            all xyz = np.concatenate((gcat_xyz,rcat_xyz))
            bbox = kszx.BoundingBox(all_xyz, pixsize=10, rpad=200)
        """

        points = utils.asarray(points, 'BoundingBox constructor', 'points', dtype=float)
        pixsize = float(pixsize)
        rpad = float(rpad)

        assert points.ndim ==2
        npoints, ndim = points.shape

        assert 1 <= ndim <= 3
        assert pixsize > 0
        assert rpad >= 0

        r2 = np.sum(points**2, axis=1)
        self.rmin = np.min(r2)**0.5
        self.rmax = np.max(r2)**0.5
        del r2

        # These arrays are shape (3,)
        xyz_min = np.min(points, axis=0)
        xyz_max = np.max(points, axis=0)
        cpos = (xyz_min + xyz_max) / 2.
        bs = (xyz_max - xyz_min)    # boxsize, before any padding

        # These arrays are shape (3,)
        self.npix_prepad = np.array(np.ceil(bs / pixsize), dtype=int)
        self.npix_preround = np.array(np.ceil((bs+rpad) / pixsize), dtype=int)
        npix = 4 * self.npix_preround  # sentinel

        # Round up to "FFT-friendly" npix values.
        # FIXME (low-priority): move this code to its own function in kszx.utils.
        
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
            print(f'    rmin = {self.rmin:.02f}, rmax = {self.rmax:.02f},', file=f)
            print(f'    npix_prepad = {self.npix_prepad},', file=f)
            print(f'    npix_preround = {self.npix_preround}', file=f)
            print(f')', file=f)
            return f.getvalue()

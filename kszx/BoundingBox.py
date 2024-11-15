import io   # StringIO
import numpy as np

from . import utils

from .Box import Box


class BoundingBox(Box):
    def __init__(self, points, pixsize, rpad):
        r"""Subclass of Box whose size/position are chosen to match a galaxy survey.

        Constructor args:

           - points (numpy array with shape (ngal,3)):
               Contains galaxy locations, in coordinate system with observer at origin.
               Usually obtained by calling Catalog.get_xyz(cosmo) -- see example below.

           - pixsize (float):
               Pixel side length (caller-specified units, usually Mpc).

           - rpad (float):
               Box padding in same units as 'points' (e.g. rpad=200 for 200 Mpc).

        The BoundingBoxconstructor will automatically choose the number of pixels (npix)
        and box position (cpos), so that the box contains the galaxy survey with the
        specified padding (rpad).

        For more info on Boxes, see the :class:`~kszx.Box` base class docstring.

        Inherits the following members from :class:`~kszx.Box`:
        
           - ndim (integer): Number of dimensions (usually 3).
           - pixsize (float): Pixel side length in user-defined coordinates.
           - boxsize (length-ndim array): Box side lengths in user-defined coordinates (equal to npix * pixsize).
           - npix (length-ndim array): Real-space map shape, represented as numpy array.
           - real_space_shape (tuple): same as 'npix', but represented as tuple instead of 1-d array
           - nk (length-ndim array): Fourier-space map shape, represented as numpy array.
           - fourier_space_shape (tuple) same as 'nk', but represented as tuple instead of 1-d array
           - cpos (length-ndim array): location of box center, in observer coordinates
           - lpos (length-ndim array): location of lower left box corner, in observer coordinates
           - rpos (length-ndim array): location of upper right box corner, in observer coordinates
           - kfund (length-ndim array): Lowest frequency on each axis (equal to 2pi/boxsize)
           - knyq (float): Nyquist frequency (equal to pi/pixsize)
           - box_volume (float): Box volume (equal to prod(boxsize))
           - pixel_volume (float): Pixel volume (equal to pixsize^N)

        The BoundingBox subclass also contains the following members:

           - rpad (float): padding that was specified in BoundingBox constructor
           - rmin (float): min distance between observer and galaxies (specified in constructor)
           - rmax (float): max distance between observer and galaxies (specified in constructor)

        Example::

            # Make bounding box which contains galaxies + randoms.
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
        self.rpad = rpad

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
            print(f'    rpad = {self.rpad:.02f},', file=f)
            print(f'    rmin = {self.rmin:.02f}, rmax = {self.rmax:.02f},', file=f)
            print(f'    npix_prepad = {self.npix_prepad},', file=f)
            print(f'    npix_preround = {self.npix_preround}', file=f)
            print(f')', file=f)
            return f.getvalue()

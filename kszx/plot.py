"""This source file is intended to be a home for plotting-related utility functions, but there's not much here yet!"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm


def real_space_plotter(arr, filename=None, axis=None, vmax=None, title=None):
    r"""General purpose function for plotting a real-space field by averaging over specified axes.
    
    Function args:
       - ``arr`` (array-like): the real-space field to plot
       - ``filename`` (str, optional): filename for saving the plot as a PDF
       - ``axis`` (list of ints, optional): the axes to average over for each plot
       - ``vmax`` (float, optional): maximum color scale value; if None, determined automatically
       - ``title``: (str, optional): title for the plot

    This function displays and optionally saves the plot with symmetric color scaling around zero (from -vmax to vmax).

    (Source: Selim's ``helperfunctions.py``, Nov 2024.)
    """

    arr = np.asarray(arr)
    
    if arr.dtype != float:
        raise RuntimeError(f'kszx.plot.real_space_plotter: expected floating-point array, got {arr.dtype=}')
    
    if axis is None:
        axis = range(arr.ndim)

    axis = [ int(x) for x in axis ]  # convert to list of ints
    assert all(0 <= d < arr.ndim for d in axis)
        
    # Initialize figure and axes for subplots, with a number of columns equal to the number of axes specified
    fig, axs = plt.subplots(1, len(axis), figsize=(4 * len(axis), 4), sharey=True, sharex=True)
    
    # Compute the mean of the real part of the matrix along each specified axis for plotting
    to_plot = [arr.mean(axis=ax) for ax in axis]
    
    # Set vmax for color scale if not provided
    if vmax is None:
        vmax = max(np.max(np.abs(x)) for x in to_plot)

    if title is not None:
        plt.title(title)
    
    # Generate images on each subplot for the averaged data along each specified axis
    im = [axs[ax].imshow(to_plot[ax], cmap=cm.RdBu_r, vmax=vmax, vmin=-vmax) for ax in range(len(axis))]

    # Add color bar to the right of the last axis (assumed to be 2)
    divider = make_axes_locatable(axs[-1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im[-1], cax=cax, orientation='vertical')
        
    # Adjust layout to avoid overlaps
    plt.tight_layout()

    # Save or show plot (depending on whether filename is None)
    if filename is not None:
        print(f'Writing {filename}')
        plt.savefig(filename)
    else:
        plt.show()

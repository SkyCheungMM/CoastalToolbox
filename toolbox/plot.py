"""
Plotting module
"""

from io import BytesIO
import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.colors import Normalize, BoundaryNorm, Colormap
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from PIL import Image
from rasterio.transform import array_bounds, from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
import requests

from .geoprocessing import coords_reproj
from .math import *

def fetch_wmts(ax:Axes, xlim:tuple, ylim:tuple, crs="EPSG:4326", 
                highres=False, **kwargs):
    """
    Fetch and add web map tiles based on x and y limits to existing plot.
    Additional keyword arguments will be used to adjust the map tiles in the 
    plot.

    Parameters
    ----------
    ax : mpl.axes.Axes
        Axis of which the plot will be casted to
    xlim, ylim : tuple
        X- and Y- limits of the current axes
    crs : str, default "EPSG:4326"
        Map projection in "EPSG:XXXX" format
    highres : bool, default False
        If true, fetch higher resolution basemap by zooming in more
    """

    # Note: the convention here is 
    # src - Incoming web map tiles from online database
    # dst - Requested projection by user
    src_crs = "EPSG:4326"

    # Default magnification
    mag = 2 if highres else 1

    # ==========================================================================
    # HELPER FUNCTIONS
    # ==========================================================================

    def num2deg(xtile, ytile, zoom):
        """Converts tile index to longitude/latitude"""
        n = 2 ** zoom
        lon = xtile/n*360 - 180
        lat = np.atan(np.sinh(np.pi*(1 - 2*ytile/n)))/np.pi*180
        return lon, lat
    
    def deg2num(lon, lat, zoom):
        """Converts longitude/latitude to tile index"""
        n = 2 ** zoom
        xtile = int((lon + 180)/360*n)
        ytile = int((1 - np.arcsinh(np.tan(lat/180*np.pi))/np.pi)/2*n)
        return xtile, ytile
    

    # ==========================================================================
    # PREPROCESSING
    # ==========================================================================

    # Get list of reprojected viewport corners in source projection (4326)
    X, Y = np.meshgrid(xlim, ylim)
    dst_corners = np.c_[X.ravel(), Y.ravel()]
    src_corners = coords_reproj(dst_corners, crs, src_crs)
    src_xlim, src_ylim = zip(src_corners.min(axis=0), src_corners.max(axis=0))

    # Estimate zoom level
    # The algorithm below obtains the maximum zoom level that has at least 2
    # independent tiles in each direction (i.e. 4 minimum total). More tiles
    # might be fetched if the viewport is elogated or has a weird shape
    for zoom in range(0, 22, 1):
        dx, dy = np.array(deg2num(src_xlim[1], src_ylim[0], zoom)) - \
                    np.array(deg2num(src_xlim[0], src_ylim[1], zoom))
        if min(dx, dy) >= mag: 
            break
    
    # Now find the tile indices and coordinates of boundaries in both directions
    xtile, ytile = deg2num(src_xlim[0], src_ylim[1], zoom)
    coords = (*num2deg(xtile, ytile+dy+1, zoom), *num2deg(xtile+dx+1, ytile, zoom))
    

    # ==========================================================================
    # DATA FETCHING
    # ==========================================================================

    # Indexing fetched images
    xindex = [None] + [1024*x for x in range(1, dx+1)] + [None]
    yindex = [None] + [1024*x for x in range(1, dy+1)] + [None]
    
    # Pull 1024x1024 web map tiles from online host (MapBox)
    # Might not work if API key has expired
    # Finally rearrange data into source matrix
    src = np.zeros((3, 1024*(dy+1), 1024*(dx+1)), np.uint8)
    for yi, yt in enumerate(range(ytile, ytile+dy+1)):
        for xi, xt in enumerate(range(xtile, xtile+dx+1)):
            url = "https://api.mapbox.com/styles/v1/mapbox/satellite-v9/tiles/" \
                + "%d/%d/%d@2x?"%(zoom, xt, yt) \
                + "access_token=pk.eyJ1Ijoic29yamFpOTUyNyIsImEiOiJjbTYwOGQ1bD" \
                + "UwYjhyMnFvdW9lajh4NGQ3In0.Grq5KxiNFLrruzRSrk2QAA"
            html = requests.get(url)
            bm = np.moveaxis(np.array(Image.open(BytesIO(html.content))), 
                                     (0,1,2), (1,2,0))
            src[:,yindex[yi]:yindex[yi+1],xindex[xi]:xindex[xi+1]] = bm


    # ==========================================================================
    # DATA PROCESSING
    # ==========================================================================

    # Define source and destination transform (affine) matrices
    src_transform = from_bounds(*coords, 1024*(dx+1), 1024*(dy+1))
    dst_transform, width, height = calculate_default_transform(src_crs,
                                    crs, 1024*(dx+1), 1024*(dy+1), *coords)
    
    # Reproject tiles
    dst = np.zeros((3, height, width), dtype=np.uint8)
    reproject(src, dst, src_transform=src_transform, src_crs=src_crs,
              dst_transform=dst_transform, dst_crs=crs,
              resampling=Resampling.nearest, dst_nodata=255)
    
    # Obtain bounds of the reprojected tiles
    dst_bounds = array_bounds(height, width, dst_transform)
    dst_bounds = (dst_bounds[0], dst_bounds[2], dst_bounds[1], dst_bounds[3])
    
    # Add to plot
    ax.imshow(np.moveaxis(dst, (0,1,2), (2,0,1)), extent=dst_bounds, **kwargs)

    return None

def plot_rose(ax:Axes, rad:np.ndarray, ang:np.ndarray, rad_bin_edge=None, 
    ang_nbins=16, arrow_dir=None, legend_title="Legend", fmt="%.01f"):
    """
    Plots a rose diagram for input data.

    Parameters
    ----------
    ax : mpl.axes.Axes
        Axis of which the plot will be casted to. The axis handle must have
        a polar projection (i.e. call ax = fig.add_subplot(projection="polar"))
    rad : numpy.ndarray
        1-D array of radial variables
    ang : numpy.ndarray
        1-D array of angular variables. Values should be specified in degrees 
        and in nautical convention (0 at North and increases clockwise)
    rad_bin_edge : None or numpy.ndarray, default None
        Bin edges for the radial variable. Defaults to quartiles if unspecified
    ang_nbins : int, default 16
        Number of equally spaced angular bins
    arrow_dir : None or str, default None
        Draw arrows indicating direction of flow. Available option are "from",
        "to" and None
    legend_title : str, default "Legend"
        Legend title
    fmt : str, default "%.01f"
        Format applied to radial bin edges in the legend
    """

    # The following features have not been implemented
    # Exclude data under a certain threshold (calms)

    # Helper functions
    def find_nearest(x, y, target):
        """
        Returns the value with cumulative probability closest to the target
        """

        idx = (np.abs(y - target)).argmin()

        return x[idx]

    # Housekeeping
    buffer_size = 0.1
    arrow_size = 0.03
    cardinals = ['N', "NNE", "NE", "ENE", 'E', "ESE", "SE", "SSE",
                 'S', "SSW", "SW", "WSW", 'W', "WNW", "NW", "NNW"]

    # Check input
    assert (len(np.array(rad).shape) == 1) & (len(np.array(ang).shape) == 1), \
        "Data must be one-dimensional.\n"
    assert (rad_bin_edge is None) | (len(np.array(rad_bin_edge).shape) == 1), \
        "Input for 'rad_bin_edge' must be one dimensional.\n"
    assert arrow_dir in [None, "from", "to"], "Incorrect input for parameter " +\
        "'arrow_dir'. Available options are 'None', 'from' and 'to' only.\n"


    # ==========================================================================
    # DATA PREPROCESSING
    # ==========================================================================

    # Keep non-nan data only and wrap angular data to 0-360 degrees
    ind = np.isnan(rad) | np.isnan(ang)
    r, t = rad[~ind], ang[~ind]%360
    
    # Compute radial bin edges for the dataset
    # Use quartiles if not specified
    if rad_bin_edge is None:
        qt, pt = ecdf(r)
        pbin = [0.25, 0.50, 0.75]   
        rbinedge = np.array([np.round(find_nearest(qt, pt, p), 2) for p in pbin])
        
    else:
        rbinedge = np.sort(rad_bin_edge)
    rbinedgestr = [fmt%x for x in rbinedge]

    # Compute angular bin edges
    dth = 360/ang_nbins
    tbinedge = np.linspace(dth/2, 360-dth/2, ang_nbins)
    

    # ==========================================================================
    # DATA BINNING
    # ==========================================================================

    # Radial bins
    # bins[i-1] < x <= bins[i]
    rbins = np.digitize(r, rbinedge, right=True)
    rlabel = ["<=%s"%rbinedgestr[0]] + \
            ["(%s,%s]"%(a,b) for a,b in zip(rbinedgestr[:-1], rbinedgestr[1:])] + \
            [">%s"%rbinedgestr[-1]]
    
    # Angular bins
    tbins = np.digitize(t, tbinedge, right=True)
    tbins[tbins == len(tbinedge)] = 0
    tvec = np.arange(0, 2*np.pi, 2*np.pi/ang_nbins)
    
    # Create relative frequency pivot table - rows: angular, columns: radial
    # I prefer not using pandas but this is the cleanest approach so far
    # Add a dummy column "dummy" to let pandas count number of entries
    # Fill empty combinations with 0
    df = pd.DataFrame({'r': rbins, 't':tbins, "dummy": np.ones_like(rbins)})
    pivot = df.pivot_table(index='t', columns='r', aggfunc="count",
                    fill_value=0).values
    pivot_norm = pivot/pivot.sum()


    # ==========================================================================
    # DATA PLOTTING
    # ==========================================================================

    # Figure customisation
    cmap, norm = set_cmap_norm(mpl.colormaps["rainbow"], vmin=-0.5, 
                    vmax=len(rbinedge))
    
    # Plot data as a stacked bar chart
    base = np.zeros_like(tvec)
    for idx in range(len(rbinedge)+1):
        ax.bar(tvec, pivot_norm[:,idx], color=cmap(norm(idx)), bottom=base,
        label=rlabel[idx], width=2*np.pi/ang_nbins*0.75, zorder=3, edgecolor='k')

        base += pivot_norm[:,idx]
    
    # Set coordinate system to nautical convention
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    # Specify tick labels
    ax.set_xticks(np.pi/180*np.linspace(0,360,16,endpoint=False), 
        labels=cardinals)
    ax.set_rlabel_position(0)

    # Draw arrows at edges
    if arrow_dir is not None:

        rmax = pivot_norm.sum(axis=1).max()
        for idx in range(ang_nbins):

            # Populate arrow parameters
            kwargs = {
                'x': 2*np.pi/ang_nbins*idx,
                "dx": 0,
                "width": 0.015,
                "head_length": 0.07*rmax,
                "edgecolor": 'k',
                "facecolor": 'k',
                "zorder": 3
            }
            
            if arrow_dir == "from":
                kwargs['y'] = rmax*(1+buffer_size+arrow_size)
                kwargs["dy"] = -rmax*arrow_size
            elif arrow_dir == "to":
                kwargs['y'] = rmax*(1+buffer_size)
                kwargs["dy"] = rmax*arrow_size
            
            ax.arrow(**kwargs)

    # Draw legend
    anchor = (0.95, 1.1) if arrow_dir is not None else (1.3, 1.1)
    ax.legend(bbox_to_anchor=anchor, title=legend_title)

    ax.yaxis.set_major_formatter(PercentFormatter(1,0))


    return None

def plot_spec(figure:Figure, ax:Axes, spec:np.ndarray, sfreq:np.ndarray, 
        sdir:np.ndarray, cmap=None, norm=None):
    """
    Plots a wave energy spectrum. 

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure containing the plotting axis
    ax : matplotlib.axes.Axes
        Axis of which the plot will be casted to. The axis handle must have
        a polar projection (i.e. call ax = fig.add_subplot(projection="polar"))
    spec : numpy.ndarray
        2D Wave energy spectrum in m2/Hz-rad. The number of rows and columns 
        should match that of the frequency and directional discretisation 
        correspondly
    sfreq : numpy.ndarray
        1D frequency discretisation array in Hz
    sdir : numpy.ndarray
        1D directional discretisation array in radians. Directions should be 
        specified in nautical convention (0 at North and increases clockwise)
    cmap : matplotlib.colors.Colormap, default None
        Colour palette applied to colour the hillshade figure based on pixel values
    norm : mpl.colors.Normalize or mpl.colors.BoundaryNorm, default None
        Normalisation (mapping) function
    """
    
    # Housekeeping
    cardinals = ['N', "NNE", "NE", "ENE", 'E', "ESE", "SE", "SSE",
                 'S', "SSW", "SW", "WSW", 'W', "WNW", "NW", "NNW"]
    Tticks = np.array([20, 10, 5, 3, 2])
    Tticklabels = ["%ds"%t for t in Tticks]

    # Check input
    assert spec.shape == (len(sfreq), len(sdir)), "Incorrect spectrum input. " \
        + "Spectrum rows and columns should correspond to frequency and " \
        + "direction respectively.\n" 

    # Set default colourmap and normalisation if not specified
    if (cmap is None) | (norm is None):
        cmap, norm = set_cmap_norm(mpl.colormaps["inferno_r"])

    # Plot data as a color mesh and add colourbar
    pcm = ax.pcolormesh(sdir, sfreq, spec, cmap=cmap, norm=norm)
    cax = figure.colorbar(pcm, ax=ax)

    # Set coordinate system to nautical convention
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    # Specify tick labels
    ax.set_xticks(np.pi/180*np.linspace(0,360,16,endpoint=False),
                  labels=cardinals)
    ax.set_yticks(1/Tticks, Tticklabels)
    ax.set_ylim(0,1/1.5)
    ax.set_rlabel_position(0)

    # Colourbar labelling
    cax.ax.set_xlabel("$\\frac{m^2}{Hz-rad}$", fontsize=14)

    plt.tight_layout()

    return None

def set_cmap_norm(cmap:Colormap, ctype="continuous", vmin=None, vmax=None, 
             boundaries=None, extend="both"):
    """
    Generates colourmap and normalisation parameters to be passed to the 
    plotting function. Outputs from this function should be referenced using 
    the 'cmap' and 'norm' parameters respectively

    Parameters
    ----------
    cmap : mpl.colors.Colormap
        Colour palettes used for mapping numeric values
    ctype : str, default 'continuous'
        Colour scale type. Available options are 'continuous' and 'discrete'
    vmin, vmax : float, default None
        Values within the range [vmin, vmax] will be linearly mapped to the full]
        extent of the chosen colourmap. Valid when continuous colourmaps are used
        Defaults to the extreme values of the input dataset if unspecified
    boundaries : array-like, default None
        Monotonically increasing sequence of bin edges used in discrete colour 
        mapping
    extend : str, default 'both'
        Extend the colourmapping to include regions beyond the boundaries 
        specified. Available options are 'both', 'min', 'max' and 'neither'

    Returns
    -------
    cout : mpl.colors.Colormap
        Updated colour palette 
    norm : mpl.colors.Normalize or mpl.colors.BoundaryNorm
        Normalisation (mapping) function 
    """

    assert ctype in ["continuous", "discrete"], "Incorrect ctype input. " +\
        "Available options are 'continuous' and 'discrete' only. \n"
    assert extend in ["both", "min", "max", "neither"], "Incorrect extend " +\
        "input. Available options are 'both', 'min', 'max' and 'neither' only. \n"

    cout = cmap.copy()

    if ctype == "continuous":
        norm = Normalize(vmin=vmin, vmax=vmax)

    elif ctype == "discrete":
        assert boundaries is not None, "Input for 'boundaries' missing. \n"
        norm = BoundaryNorm(boundaries, 256, extend=extend)

    if extend in ["min", "neither"]: cout.set_over(alpha=0)
    if extend in ["max", "neither"]: cout.set_under(alpha=0)

    return cout, norm

def show_mpl_cmap():
    """
    Plot all colourmaps included in matplotlib
    """

    cmap_full_list = [('Perceptually Uniform Sequential', [
        'viridis', 'plasma', 'inferno', 'magma', 'cividis']),
        ('Sequential', [
        'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
        'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
        'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),
        ('Sequential (2)', [
        'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
        'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
        'hot', 'afmhot', 'gist_heat', 'copper']),
        ('Diverging', [
        'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
        'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']),
        ('Cyclic', ['twilight', 'twilight_shifted', 'hsv']),
        ('Qualitative', [
        'Pastel1', 'Pastel2', 'Paired', 'Accent',
        'Dark2', 'Set1', 'Set2', 'Set3',
        'tab10', 'tab20', 'tab20b', 'tab20c']),
        ('Miscellaneous', [
        'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
        'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
        'gist_rainbow', 'rainbow', 'jet', 'turbo', 'nipy_spectral',
        'gist_ncar'])]

    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))

    tot = []
    for cmap_list in cmap_full_list:
        tot.extend(cmap_list[1])

    fig = plt.figure()
    fig.suptitle("List of inbuilt colour maps (add suffix '_r' for colour reversal)")

    manager = plt.get_current_fig_manager()
    #manager.window.showMaximized()

    n_rows = 4
    for i in range(len(tot)):
        ax = fig.add_subplot(len(tot)//n_rows+1,n_rows,i+1)
        ax.imshow(gradient, aspect='auto',cmap=tot[i])
        ax.text(-5,0.5,tot[i],va='center',ha='right')
        ax.set_axis_off()

    plt.subplots_adjust(left=0.075,right=0.975,bottom=0.05,top=0.9)
    plt.show()

    return None


    
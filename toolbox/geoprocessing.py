"""
Geoprocessing module
"""

from contourpy import contour_generator
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, BoundaryNorm, Colormap
from matplotlib.path import Path
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.transform import from_origin, xy
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import KDTree
from shapely.geometry import *

def _calc_hillshade(surf:np.ndarray, dx:float, dy:float, altitude:float, 
                   azimuth:float, zfactor:float):
    """
    Calculates the hypothetical illumination of a surface. The surface must be 
    defined on a grid with regular grid spacing.

    Parameters
    ----------
    surf : numpy.ndarray
        Surface elevation with (i,j) = (y,x)
    dx : float
        Spacing along the x-direction
    dy : float
        Spacing along the y-direction
    altitude : float
        Elevation angle of the light source in degrees
    azimuth : float
        Light source orientation in degrees True North
    zfactor : float
        Feature elevation exaggeration factor
    
    Returns
    -------
    shade : numpy.ndarray
        Pixel output colour depth from 0 (black) to 255 (white)
    """

    # Compute zenith and azimuth angles
    theta_z = (90 - altitude)*np.pi/180
    theta_az = (360 - azimuth + 90)*np.pi/180

    # Rows have same y, columns have same x
    grad_y, grad_x = np.gradient(surf, dy, dx)
    grad = np.arctan2(np.sqrt(grad_x**2 + grad_y**2)*zfactor, 1)
    aspect = np.arctan2(grad_y, -grad_x)

    # Compute the shading value, and set minimum to 0
    shade = 255*(np.cos(theta_z)*np.cos(grad) + \
                 np.sin(theta_z)*np.sin(grad)*np.cos(theta_az - aspect))
    
    shade = np.maximum(shade, np.zeros_like(shade))

    return shade

def contour_from_grid(X:np.ndarray, Y:np.ndarray, Z:np.ndarray, levels:list):
    """
    Compute coordinates of contours based on a gridded elevation dataset at 
    given levels

    Parameters
    ----------
    X, Y : numpy.ndarray
        X- and Y-coordinates of the grid. Input should have the same dimensions
        as the Z values 
    Z : numpy.ndarray
        The 2D gridded value to calculate the contours of. May be a masked array,
        and any invalid values (np.inf or np.nan) will be masked out
    levels : list or array-like
        Z-levels to calculate contours at

    Returns
    -------
    levels_asc : numpy.ndarray
        Unique levels within range of the Z dataset, sorted in ascending order
    contours : list
        Vertices of the contours. contour[0] gives the vertices of contours 
        corresponding to the first level, contours[1] the second level etc
    """

    # Presort levels in ascending order
    # Retain levels within limits of the dataset
    zmin, zmax = np.nanmin(Z), np.nanmax(Z)
    levels_asc = np.sort(np.unique(levels), None)
    levels_asc = levels_asc[(zmin <= levels_asc) & (levels_asc <= zmax)]

    # Compute contours 
    cont_gen = contour_generator(X, Y, Z)
    contours = cont_gen.multi_lines(levels_asc)

    return levels_asc, contours

def coords_reproj(xy:np.ndarray, crs_from:str, crs_to:str):
    """
    Reproject coordinates to the specified projection

    Parameters
    ----------
    xy : numpy.ndarray
        Input coordinates with shape (N,2), columns being x- and y-coordinates 
        respectively
    crs_from : str
        Current projection in 'EPSG:XXXX' format
    crs_to : str
        Target projection in 'EPSG:XXXX' format
    
    
    Returns
    -------
    xy_reproj : numpy.ndarray
        Coordinates in the target projection
    """

    pts = np.array(xy)

    # Check input dimension
    assert pts.shape[1] == 2, "Incorrect input dimensions. Input coordinates " +\
        "should have two columns only.\n"

    # Read projection details 
    src_crs = pyproj.CRS(crs_from)
    dst_crs = pyproj.CRS(crs_to)

    # Do not perform reprojection if input and output projections are the same
    if src_crs == dst_crs: return pts
    
    # Set up transformer and reproject points
    transformer = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    xy_reproj = transformer.transform(pts[:,0], pts[:,1])
    xy_reproj = np.array(xy_reproj).T

    return xy_reproj

def grid_to_raster(dst:str, coords:np.ndarray, crs:str, fill=np.nan):
    """
    Creates a raster from gridded points. No interpolation is carried out
    
    Parameters
    ----------
    dst : str
        Output raster path
    coords : numpy.ndarray
        Array of grid points. Input should have dimensions of (N, 3), with rows
        representing individual points and columns representing X, Y, Z 
        coordinates respectively
    crs : str
        Projection code in the format of 'EPSG:XXXX'
    fill : float, default np.nan
        Filler value for regions without data
    """

    coords = np.array(coords)

    assert (coords.ndim == 2) & (coords.shape[1] == 3), "Incorrect coordinate " +\
        "input. Please refer to input description. \n"
    
    df = pd.DataFrame(data={'X': coords[:,0], 'Y': coords[:,1], 'Z': coords[:,2]})

    # Compute unique values along the X- and Y-directions and set up grid 
    X_unique = df.X.unique(); X_unique.sort()
    Y_unique = df.Y.unique(); Y_unique.sort()
    dx = X_unique[1] - X_unique[0]
    dy = Y_unique[1] - Y_unique[0]
    lx = len(X_unique); ly = len(Y_unique)

    # Create pivot table
    df_pivot = df.pivot_table(columns='X', index='Y', values='Z', fill_value=fill)

    # Set up raster information and georeference
    transform = from_origin(X_unique[0]-dx/2, Y_unique[-1]+dy/2, dx, dy)
    meta = {
        "driver": "GTiff",
        "width": lx,
        "height": ly,
        "count": 1,
        "dtype": str(df.Z.dtype),
        "crs": crs,
        "transform": transform
    }

    # Export raster
    with rasterio.open(dst, 'w', **meta) as dst:
        dst.write(np.flipud(df_pivot.values), 1)
        dst.close()

    return None

def hillshade(src: str, dst=None, hs_type="traditional", zfactor=1.0, 
              nodata=None, altitude=45.0, azimuth=315.0, md_method=1,
              cmap=None, norm=None, cbar_label=None, transparency=0.3, 
              cbar_kwargs=None):
    """
    Computes the hypothetical illumination of a terrain and produces a 3D 
    representation (shade) by taking into account the sun's relative position
    while shading the image.
    
    Parameters
    ----------
    src : str
        Source elevation raster path. Only the first band will be read
    dst : str, default None
        Destination of the hillshade raster. If this parameter is unspecified,
        the default output location will be in the same directory as the source
        raster file but with '_hs' appended to the file name
    hs_type : str, default 'traditional'
        Hillshade figure type. Available options are 'traditional', 
        'traditional_coloured', 'multidirectional', and 'multidirectional_coloured'
    zfactor : float, default 1
        Feature elevation exaggeration factor used in hillshade computation
    nodata : float, default None
        Placeholder value indicating pixel has no data and should be ignored
        during calculations. Assumes np.inf is the placeholder value if left
        unspecified
    altitude : float, default 45
        Elevation angle of the light source in degrees
    azimuth : float, default 315
        Light source orientation in degrees True North
    md_method : int, default 1
        Weighting scheme applied to different illumination sources upon computing
        the multidirectional hillshade. Available options are 1 and 2 only. 
        Details can be found at:
        1. https://www.staridasgeography.gr/shadowplay/
        2. https://community.esri.com/t5/arcgis-pro-questions/multidirectional-hillshade-with-custom-main/td-p/1204254
    cmap : mpl.colors.Colormap or None
        From set_cmap_norm() function outputs. Colour palette applied to 
        colour the hillshade figure based on pixel values
    norm : mpl.colors.Normalize or mpl.colors.BoundaryNorm or None
        From set_cmap_norm() function outputs. Normalisation (mapping) function 
        applied to the pixel values
    cbar_label : str, default None
        Colourbar label
    transparency : float, default 0.3
        Colouring transparency. Ranges from 0 (colour scheme is fully transparent)
        to 1 (colour scheme is fully opaque)
    cbar_kwargs : dict, default None
        Additional parameters to support colourbar construction
    """

    # Multidirectional layers
    layers = {
        1: {
            "Altitude": [34, 34, 25, 80, 25],
            "Azimuth": [312, 48, 110, 180, 250],
            "Weighting": [0.5, 0.1, 0.1, 0.4, 0.25]
            },

        2: {
            "Altitude": [70, 60, 55],
            "Azimuth": [350, 15, 270],
            "Weighting": [0.65, 0.5, 0.7]
        }
    }

    # Input check
    assert hs_type in ["traditional", "traditional_coloured", "multidirectional",
                       "multidirectional_coloured"], "Incorrect hs_type input. \n"
    assert md_method in [1, 2], "Incorrect method input. Available options are" +\
        " 1 and 2 only. \n"
    
    # Transparency
    alpha = 0 if "coloured" not in hs_type else 1 - transparency
    
    # Output path
    if dst is None:
        fpout = src.split('.')
        fpout[-2] += "_hs"
        fpout = '.'.join(fpout)
    
    else: fpout = dst

    # Read source file and obtain grid spacing
    raster = rasterio.open(src)
    dx = np.abs(raster.transform[0]); dy = np.abs(raster.transform[4])

    # Calculate grayscale hillshade values and mask array
    surf = raster.read(1); surf[np.isinf(surf)] = np.nan
    if nodata is not None: surf[surf == nodata] = np.nan

    if "traditional" in hs_type:
        hs = _calc_hillshade(surf, dx, dy, altitude, azimuth, zfactor)

    else: 
        hs = [w*_calc_hillshade(surf, dx, dy, alt, azi, zfactor) for (alt, azi, w)
              in zip(layers[md_method]["Altitude"], layers[md_method]["Azimuth"],
                     layers[md_method]["Weighting"])]
        hs = np.sum(hs, axis=0)/np.sum(layers[md_method]["Weighting"])
    hs_masked = np.ma.array(hs, mask=np.isnan(surf))
    fg_masked = np.ma.array(surf, mask=np.isnan(surf))

    # Background colourmap
    # Set transparency to 100% at locations without data
    bg_cmap = mpl.colormaps["gray"]
    bg_cmap.set_bad(alpha=0)

    # Normalise colours
    bg_norm = Normalize(vmin=0, vmax=255)

    # Compute background RGBA values
    bg_rgba = bg_cmap(bg_norm(hs_masked))

    # Foreground colouring
    if "coloured" in hs_type:
        assert isinstance(cmap, Colormap), "Incorrect cmap input. Only " +\
            "matplotlib.colors.Colormap instances are accepted. \n"
        fg_cmap, fg_norm = cmap.copy(), norm
        fg_cmap.set_bad(alpha=0)

        fg_rgba = fg_cmap(fg_norm(fg_masked))

        # Plot colourbar
        cbout = fpout.split('.')
        cbout[-2] += "_cbar"
        cbout = '.'.join(cbout)
        cbout = cbout.replace(".tif", ".png")

        cbar_kwargs = cbar_kwargs if isinstance(cbar_kwargs, dict) else {}
        hillshade_colourbar(cbout, fg_cmap, fg_norm, cbar_label, **cbar_kwargs)

    else: fg_rgba = np.zeros_like(bg_rgba)

    # Blend colours with 100% opacity for background image
    merge_rgba = bg_rgba*(1-alpha) + fg_rgba*alpha
    merge_alpha = merge_rgba[:,:,-1].reshape((*merge_rgba.shape[:2], 1))
    merge_rgba = np.divide(merge_rgba, merge_alpha, where=merge_alpha != 0)
    merge_rgba = (255*merge_rgba).astype(np.uint8)

    # Output hillshade as a 8-bit (4-channel) raster
    # Create raster using similar metadata inputs as the source file
    meta = raster.meta.copy()
    meta.update({"nodata": 0, "count": 4, "dtype": str(merge_rgba.dtype)})

    # Reorder dimensions and write output file
    with rasterio.open(fpout, 'w', **meta) as f:
        f.write(np.moveaxis(merge_rgba, [0,1,2], [1,2,0]))
        f.close()

    return None

def hillshade_colourbar(dst, cmap, norm, label, figsize=(1,3), **kwargs):
    """
    Plots the colourbar associated with the palette used to produce the 
    coloured hillshade. Additional keywords are fed to 'plt.colorbar()' 

    Parameters
    ----------
    dst : str
        Destination of the colourbar figure
    cmap : matplotlib.colors.Colormap
        Colour palette applied to colour the hillshade figure based on pixel values
    norm : mpl.colors.Normalize or mpl.colors.BoundaryNorm 
        Normalisation (mapping) function
    label : str
        Colourbar label
    figsize : tuple, default (1,3)
        Figure size in inches
    """

    # Default colourbar keywords
    cbar_kwargs = {"orientation": "vertical"}
    cbar_kwargs.update(kwargs)

    # Initiate figure and plot colourbar
    fig, ax = plt.subplots(figsize=figsize)
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=ax, label=label, 
                 **cbar_kwargs)
    ax.yaxis.set_label_position("left")
    plt.tight_layout()

    # Save colourbar
    fig.savefig(dst)

    return None

def intersect_segment(L:np.ndarray, P:np.ndarray):
    """
    Extract the segment <u>closest</u> to the specified point

    Parameters
    ----------
    L : numpy.ndarray
        List of coordinates representing the vertices of the polyline. Input
        should have dimensions of (N,2) with columns representing x- and y- 
        coordinates respectively
    P : numpy.ndarray
        Query coordinates. Input should be 1D and have shape (2,)
    
    Returns
    -------
    l : float
        Distance from the first vertex to the specified point along the polyline
    S : numpy.ndarray
        Segment containing the intersection coordinates in [[xi,yi], [xi+1, yi+1]]
    """

    L = np.array(L)
    P = np.array(P).ravel()

    assert (L.ndim == 2) & (L.shape[1] == 2), "Incorrect vertex coordinate " +\
        "input. Please refer to input description. \n"
    
    ls = LineString(L)
    pt = Point(P)

    # Compute distance to point
    l = ls.line_locate_point(pt)

    # Cumulative distance at vertices
    delta = np.sqrt(np.sum((L[1:,:] - L[:-1,:])**2, axis=1))
    cumdist = np.insert(np.cumsum(delta), 0, 0)

    # Find intersection segment
    idx = [np.argwhere(cumdist < l).max(), np.argwhere(l < cumdist).min()]
    S = np.vstack((L[idx[0],:], L[idx[1],:]))

    return l, S

def linear_interp(xyz:np.ndarray, query:np.ndarray, max_dist=None):
    """
    Performs linear interpolation for the query points based on data provided.
    Note interpolation only takes place within the convex hull of the dataset.

    Parameters
    ----------
    xyz : numpy.ndarray
        Array of data point coordinates. Input should have shape of (N,3) with
        columns representing values in the x-, y- and z-direction respectively
    query : numpy.ndarray
        Array of query point coordinates. Input should have shape of (N,2) with 
        columns representing x- and y-coordinates respectively.
    max_dist : float, default None
        Maximum distance from data points for the interpolated values to be
        considered valid

    Returns
    -------
    Zinterp : numpy.ndarray
        Interpolated elevation 
    """

    # Convert to array
    xyz = np.array(xyz)
    query = np.array(query)

    # Check input dimensions
    assert (xyz.shape[1] == 3) & (query.shape[1] == 2), "Incorrect input " +\
        "dimensions. Please check input description.\n"
    
    # Perform linear interpolation
    interp = LinearNDInterpolator(xyz[:,:2,], xyz[:,2])
    Zinterp = interp(query[:,0], query[:,1])

    # Limit values if maximum distance is specified
    # Create a KD-Tree to filter out points far from existing data
    if max_dist is not None:
        tree = KDTree(xyz[:,:2])
        dist, _ = tree.query(query, workers=-1)
        Zinterp[dist > max_dist] = np.nan

    return Zinterp

def raster2array(src:rasterio.DatasetReader, fill=None):
    """
    Converts the input raster to array (long format). Filler values for regions 
    without data will be set to np.nan. All bands will be extracted as columns 
    in the exact same order.
    
    Parameters
    ----------
    src : rasterio.DatasetReader
        Raster dataset obtained from rasterio.open()
    fill : int or float, default None
        Filler values for regions without data. 
    
    Returns
    -------
    xyz : numpy.ndarray
        Coordinates and z values of the raster
    """

    # Compute grid dimensions
    height, width = np.arange(src.height), np.arange(src.width)

    # Transform grid to source file coordinates
    rows, cols = np.meshgrid(height, width, indexing="ij")
    xt, yt = xy(src.transform, rows, cols)

    # Now read elevation values for all bands
    # Also set fill values to nan
    z = src.read()
    if z.dtype != np.uint8:
        if fill is None: z[np.isinf(z)] = np.nan
        else: z[z==fill] = np.nan

    # Reshape into long format
    xyz = np.c_[np.reshape(xt, (-1,1)),
                np.reshape(yt, (-1,1)),
                np.reshape(np.moveaxis(z, (0,1,2), (2,0,1)), (-1,src.count))]
    
    return xyz

def raster_reproj(src:str, dst=None, crs="EPSG:4326"):
    """
    Reprojects the raster to the specified projection. Resampling will be done 
    on the basis of nearest neighbour

    Parameters
    ----------
    src : str
        Source raster the requires reprojection
    dst : str, default None
        Destination of the reprojected raster. If this parameter is unspecified, 
        the default output location will be in the same directory as the source 
        raster file but with '_reproj' appended to the file name
    crs : str, default 'EPSG:4326'
        Target projection in 'EPSG:XXXX' format
    """

    # Output path
    if dst is None:
        fpout = src.split('.')
        fpout[-2] += "_reproj"
        fpout = '.'.join(fpout)
    
    else: fpout = dst

    # Open file and compute transformation and transformed dimensions
    raster = rasterio.open(src, 'r')
    transform, width, height = calculate_default_transform(raster.crs, crs,
                                raster.width, raster.height, *raster.bounds)
    kwargs = raster.meta.copy()
    kwargs.update({
        "crs"       : crs,
        "transform" : transform,
        "width"     : width,
        "height"    : height
    })

    # Write reprojected raster
    rasterout = rasterio.open(fpout, 'w', **kwargs)
    for i in range(1, raster.count+1):
        reproject(
            source          = rasterio.band(raster, i),
            destination     = rasterio.band(rasterout, i),
            src_transform   = raster.transform,
            src_crs         = raster.crs,
            dst_transform   = transform,
            dst_crs         = crs,
            resampling      = Resampling.nearest
        )

    raster.close()
    rasterout.close()

    return None

def select_by_location(vertices:np.ndarray, X:np.ndarray):
    """
    Select features located within the polygon defined using vertices provided.

    Parameters
    ----------
    vertices : numpy.ndarray
        Vertices of the bounding polygon. Should have shape of (N,2), with 
        columns representing coordinates in the x- and y-directions respectively
    X : numpy.ndarray
        Data coordinates. Input should have shape of (N,2) with columns 
        representing x- and y-coordinates respectively.

    Returns
    -------
    b : numpy.ndarray
        Boolean array of length N indicating if a point is within the polygon
    """

    # Convert to array
    vertices = np.array(vertices)
    X = np.array(X)

    # Check input dimensions
    assert (vertices.shape[1] == 2) & (X.shape[1] == 2), "Incorrect input " +\
        "dimensions. Inputs should have two columns. \n"
    
    # Create polygon
    bounds = Path(vertices)

    # Check if points are located within the polygon
    b = bounds.contains_points(X)

    return b



    
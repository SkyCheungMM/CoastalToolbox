"""
Math module
"""

import numpy as np

def boxcar(y, span, nanTreat=False, endTreat=True):
    """
    Run a boxcar filter through the signal 

    Notes
    -----
        - Boxcar running average cutoff frequency is found by
            fc = 0.6/(time_span)
        - Double running average filter:
            fc = 0.4429/(time_span)
        - Operates on first dimension of the array
        - A lot of assumptions about the data are made here, this function is by
            no means as robust as Matlab's smooth function. Only real valued
            numbers are assumed to be passed to the array and no repetition in the
            coordinate variable is assumed. Use at your own risk.

    Parameters
    ----------
    y : numpy.ndarray
        1D Signal to be filtered
    span : int
        Stencil (filter window) width in terms of number of data points. Input 
        should be an odd integer
    nanTreat : bool, default False
        Ignore nan values while computing boxcar average
    endTreat : bool, default True
        Reduce stencil width at both ends of the signal while computing 
        boxcar average
    
    Returns
    -------
    ybox : numpy.ndarray
        Boxcar filtered signal with the same length as input
    """

    # Quick data check
    if span > y.shape[0]:
        print("Stencil of " + np.str(span) + " is larger than the " +
                "length of the array (" + np.str(y.shape[0]) + ")")
        return np.nan

    # Span must be an odd number
    width = span - 1 + span % 2
    offset = np.int64((width - 1.0)/2.0)

    # Preallocate variable
    ybox = np.zeros_like(y)

    # Find indices for averaging
    first = np.int64(np.ceil(width/2.0) - 1.0)
    last = np.int64(y.shape[0] - first - 1.0)
    
    if nanTreat:
        for aa in range(first,last+1):
            if ~np.isnan(y[aa]):
                tmpW = np.sum(np.isfinite(y[aa-offset:aa+offset+1]),axis=0)
                ybox[aa] = np.nansum(y[aa-offset:aa+offset+1],axis=0)/tmpW
            else:
                ybox[aa] = np.nan

    else:
        for aa in range(first,last+1):
            ybox[aa] = np.sum(y[aa-offset:aa+offset+1],axis=0)/width

    # Provide end treatment
    if endTreat:
        for aa in range(0,first):
            ybox[aa] = (np.sum(y[0:aa+offset+1],axis=0) /
                        (aa + offset + 1.0))

        for aa in range(last+1,y.shape[0]):
            ybox[aa] = (np.sum(y[aa-offset::],axis=0) /
                        (y.shape[0] - aa + offset + 0.0))

    else:
        ybox[:first] = y[:first]
        ybox[last:]  = y[last:]

    return ybox

def countTrue(b:np.ndarray):
    """
    Counts the number (length) for all consecutive True values in a boolean array

    Parameters
    ----------
    b : numpy.ndarray
        1D Boolean array

    Returns
    -------
    count : numpy.ndarray
        List of lengths of consecutive True values
    """

    # Convert to numpy array if input is of wrong format
    b = np.array(b)
    assert (len(b.shape) == 1) & (b.dtype == np.bool_), "1D boolean array " +\
            "required.\n"

    count = np.diff(np.where(np.concatenate((
                [b[0]], b[:-1] != b[1:], [True])))[0])[::2]
    
    return count

def ecdf(y:np.ndarray, decimals=3):
    """
    Compute the empirical cumulative distribution function for a dataset. 
    NaN values are neglected in calculations

    Parameters
    ----------
    y : numpy.ndarray
        1D numerical dataset
    decimals : int, default 3
        Number of decimal places to round data to
    
    Returns
    -------
    qs : numpy.ndarray
        Random variable values in ascending order
    ps : numpy.ndarray
        Cumulative probabilities of the random variables
    """

    # Convert input array to 1D, then round off values and sort
    # Also drop na values
    ysort = np.sort(np.round(np.array(y).ravel(), decimals))
    ysort = ysort[~np.isnan(ysort)]

    # Compute unique values
    qs, cs = np.unique(ysort, return_counts=True)
    ps = cs.cumsum()/(cs.sum()+1)

    return qs, ps

def zero_crossing(X:np.ndarray, shift=0):
    """
    Find peaks and troughs of a signal using the zero-upcrossing method. 
    The first and last half-open segments are not used in computation. This code 
    is not optimised for large-scale data processing and may be slow
    
    Parameters
    ----------
    eta : numpy.ndarray
        Signal to be analysed
    shift : float, default 0
        Downward shift applied to the signal. This is equivalent to drawing the 
        "zero-crossing line" at elevation=shift
        
    Returns
    -------
    idx_p : numpy.ndarray
        Indicies of values that just crossed zero and is positive
    peaks : numpy.ndarray
        Peak values
    troughs : numpy.ndarray
        Trough values
    """

    # Turn data into array format and shift
    array = np.array(X)
    array = array - shift

    # Use the sign function to help identify locations of zero-crossings
    sgn = np.sign(array)
    sgn_change = sgn[:-1]*sgn[1:]
    idx = np.where((sgn_change == -1) | (sgn_change == 0))[0]
    idx_p = idx[np.where(sgn[idx+1] > 0)]+1

    # Now find peaks and troughs. Ignore first and last segment
    assert len(idx_p) >= 2, "No segments identified."

    segments = np.split(array, idx_p)[1:-1]
    peaks = np.array(list(map(lambda x: np.max(x) + shift, segments)))
    troughs = np.array(list(map(lambda x: np.min(x) + shift, segments)))

    return idx_p, peaks, troughs



import scipy as sp
import numpy as np
import xarray as xr
from itertools import combinations
import pandas as pd
from scipy.stats import zscore
from scipy.signal import find_peaks

# Adapted from code originally provided by Josh Faskowizt
def get_ets(ts1, ts2=None, normalize=False): 
    """Creates edge timeseries
    input:     
        timeseies, size: time x channel. It can be a numpy.array or an xarray.DataArray.
         if the latter, then we will try to use the index to infer roi names and then
         edge names
    output:    
        edge timeseries, size time x |channels*(channels-1)/2|
     If only one ts is provided, we assume we are doing that to itself
    """
    roi_names  = None
    if isinstance(ts1,xr.core.dataarray.DataArray):
        if 'roi' in ts1.dims:
            roi_names = list(ts1.coords['roi'].values)
        ts1 = ts1.values
    if ts2 is None:
        ts2 = ts1
    else:
        if isinstance(ts2,xr.core.dataarray.DataArray):
            ts2 = ts2.values
   
    assert ts1.shape == ts2.shape, "++ ERROR [get_ets]: Timeseries shape do not match."
        
    # zscore the input timeseries
    if normalize:
        z1 = sp.stats.zscore(np.asarray(ts1),0)    
        z2 = sp.stats.zscore(np.asarray(ts2),0)
    else:
        z1 = ts1
        z2 = ts2
        
    T, N= ts1.shape
    u,v = np.where(np.triu(np.ones(N),1))           # get edges
    # element-wise prroduct of time series
    ets = (z1[:,u]*z2[:,v])

    if roi_names is not None:
        # Create edge names
        edge_list_tuples = [e for e in combinations(roi_names,2)]
        edge_list_idx    = ['-'.join(e) for e in edge_list_tuples]

        ets = pd.DataFrame(ets,columns=edge_list_idx)
    if normalize:
        ets.name = 'ETS (Z-Scored)'
    else:
        ets.name = 'ETS (As is)'
    return ets

def root_sum_of_squares(x):
    return np.sqrt(np.sum(x**2))

def detect_RSSevents(time_series, Nreps=1000, offsets=None, pthr=0.05):
    Ntp, Nchannels = time_series.shape
    
    if offsets is None:
        offsets = np.arange(Ntp)

    # Compute ets and rss
    ets   = fcn_ets(time_series)
    rssts = np.sqrt(np.sum(ets**2, axis=1))

    # Generate null distribution with circshift
    pcnt = np.zeros(Ntp)
    for r in range(Nreps): #tqdm(range(Nreps), desc='Repetition'):
        tsr = np.zeros((Ntp, Nchannels))
        for n in range(Nchannels):
            # Apply circshift using a random offset
            shift = np.random.choice(offsets)
            tsr[:, n] = np.roll(time_series[:, n], shift)
        
        etsr = fcn_ets(tsr)
        rsstsr = np.sqrt(np.sum(etsr**2, axis=1))
        pcnt += (rssts > np.max(rsstsr)).astype(int)
    
    # Compute p-values
    pval = 1 - pcnt / Nreps

    # Find peaks
    fp_ts, _ = find_peaks(rssts)
    
    # Intersection of peaks and significant points
    pk_inds = np.intersect1d(np.where(pval < pthr)[0], fp_ts)
    pk_amp = rssts[pk_inds]
    numpk = len(pk_inds)

    return pk_inds, pk_amp, numpk, pval, pcnt

def fcn_ets(ts):
    _, n = ts.shape
    z = zscore(ts, axis=0)  # z-score
    u, v = np.triu_indices(n, k=1)  # get edges
    a = z[:, u] * z[:, v]  # edge time series products
    
    return a
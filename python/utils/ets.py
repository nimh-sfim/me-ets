import scipy as sp
import numpy as np
import xarray as xr
from itertools import combinations
import pandas as pd
# Adapted from code originally provided by Josh Faskowizt
import scipy as sp
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
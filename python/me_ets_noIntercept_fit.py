import argparse
import os.path as osp
import xarray as xr
import numpy as np
from utils.ets import get_ets
from itertools import combinations, product
from tqdm import tqdm 

def read_command_line():
    parser = argparse.ArgumentParser(description='Perform Linear Fit across echoes on each time point. Model does not include intersect')
    parser.add_argument('-e','--echo_times', type=str, help='Echo times separated by commas', required=False, dest='tes', default=None)
    parser.add_argument('-r','--roits_path', type=str, help='Path to file with ROIts in xarray', required=True, dest='roits_path')
    parser.add_argument('-w','--out_path', type=str, help='Path to output file', required=True, dest='out_path')
    parser.add_argument('-o','--overwrite', help='Overwrite output file?', required=False, dest='overwrite', action='store_true')
    parser.add_argument('-v','--verbose', help='Print as much info as possible?', required=False, dest='verbose', action='store_true')
    parser.add_argument('-z','--zscore_roi_ts', help='Z-score each roi ts separately prior to ETS computation', required=False, dest='zscore', action='store_true')
    parser.add_argument('-m','--model',help='Model to use during fitting',required=True,choices=['ax+b','ax','ax2+bx+c'], dest='model',default=None)
    parser.set_defaults(overwrite=False)
    parser.set_defaults(verbose=False)
    parser.set_defaults(zscore=False)
    return parser.parse_args()

def main():
    opts = read_command_line()
    # Print Input Parameters
    print("++ INFO: Echo times      : %s" % opts.tes)
    print("++ INFO: ROI TS path     : %s" % opts.roits_path)
    print("++ INFO: Output path     : %s" % opts.out_path)
    print("++ INFO: Overwrite output: %s" % str(opts.overwrite))
    print("++ INFO: Verbode         : %s" % str(opts.verbose))
    print("++ INFO: Zscore          : %s" % str(opts.zscore))
    print("++ INFO: Fit Model       : %s" % str(opts.model))
    print("===============================================================================")
    # Basic input checks
    # ==================
    assert osp.exists(opts.roits_path),"++ ERROR: -----> Input ROI Timeseries not found."
    assert not (osp.exists(opts.out_path) & (not opts.overwrite)), "++ ERROR: ----> Output file exists and you did not set overwtite to true."
    # Load roi_ts
    # ===========
    print("++ INFO: Loading ROI timeseries Xarray....")
    roi_ts = xr.open_dataarray(opts.roits_path)
    if opts.verbose:
        print(roi_ts.coords) 
    assert ('te' in roi_ts.dims),   "++ ERROR: ----> 'te' is not one of the dimensions in the ROI TS xarray."
    assert ('roi' in roi_ts.dims),  "++ ERROR: ----> 'roi' is not one of the dimensions in the ROI TS xarray."
    assert ('tr' in roi_ts.dims),   "++ ERROR: ----> 'tr is not one of the dimensions in the ROI TS xarray."
    # Ensure consistent TE information if provided twice
    assert ('te' in roi_ts.dims) or (opts.tes is not None), "++ ERROR: -----> Echo Times not provided manually and not available in ROI TS xarray." 
    roi_tes_fromXR = None
    roi_tes_fromCL = None
    roi_tes        = None
    if 'te' in roi_ts.dims:
        roi_tes_fromXR = roi_ts.coords['te'].values
    if opts.tes is not None:
        roi_tes_fromCL = np.array([float(i) for i in opts.tes.split(',')])
    if (roi_tes_fromXR is not None) and (roi_tes_fromCL is not None):
        assert np.all(roi_tes_fromCL == roi_tes_fromXR), "++ ERROR: -----> TEs in ROI TS array %s and those provided manually %s do not match." % (str(roi_tes_fromXR),str(roi_tes_fromCL)) 
    if (roi_tes_fromXR is None) and (roi_tes_fromCL is not None):
        roi_tes = roi_tes_fromCL 
    elif (roi_tes_fromXR is not None) and (roi_tes_fromCL is None):
        roi_tes = roi_tes_fromXR
    else:
        roi_tes = roi_tes_fromXR
    print("++ INFO: Echo Times are: %s" % str(roi_tes))
 
    ets_tes_pairs = [i for i in product(roi_tes,repeat=2)]
    ets_tes_idx   = [str(a)+'x'+str(b) for (a,b) in ets_tes_pairs]
    ets_tes       = np.array([i*j for i,j in ets_tes_pairs]).round(2)
    print('++ INFO: ets te pairs :' , ets_tes_pairs)
    print('++ INFO: ets tes      :' , ets_tes)
    # Extract the other coordinates
    Nrois     = roi_ts.coords['te'].shape[0]
    roi_names = list(roi_ts.coords['roi'].values)
    Nacqs     = roi_ts.coords['tr'].shape[0]
    tr_idx    = list(roi_ts.coords['tr'].values)

    # Create edge names
    edge_list_tuples = [e for e in combinations(roi_names,2)]
    edge_list_idx    = ['-'.join(e) for e in edge_list_tuples]
    Nedges = len(edge_list_idx)
    print('++ INFO: Number of edges : %d' % Nedges)
    print('++ INFO: Edges : %s ...' % str(edge_list_idx[0:10]))
    
    # Compute ets
    # ===========
    print("++ Compute ETS....")
    ets = xr.DataArray(dims=['tes','tr','edge'],
                       coords={'tes':ets_tes_idx,'tr':tr_idx,'edge':edge_list_idx})
    ets.name = 'ets'

    for i in tqdm(range(len(ets_tes_pairs))):
        te_t1,te_t2 = ets_tes_pairs[i]
        aux_ts1 = roi_ts.sel(te=te_t1).values
        aux_ts2 = roi_ts.sel(te=te_t2).values
        ets.loc[str(te_t1)+'x'+str(te_t2),:,:] = get_ets(aux_ts1,aux_ts2, normalize=opts.zscore) 
    print(ets)
    
    # Create output xarray
    # ====================
    if opts.model == 'ax':
        fit_metrics = ['R2','R2_adj','F','F_pval','Slope','Slope_T','Slope_pval']
    if opts.model == 'ax+b':
        fit_metrics = ['R2','R2_adj','F','F_pval','Slope','Slope_T','Slope_pval','Intercept','Intercept_T','Intercept_pval']

    ets_fit = xr.DataArray
    # Compute model fits
    
    # Write to disk

if __name__ == "__main__":
    main()

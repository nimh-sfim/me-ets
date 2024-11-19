import holoviews as hv
from .basics import TASK_COLORS, TASKS, DATASET, SCHEDULES_DIR,  get_paradigm_info
import pandas as pd
import numpy  as np
import xarray as xr
import panel as pn
import hvplot.pandas
import holoviews as hv
from holoviews import opts, dim
import os.path as osp
from sfim_lib.plotting.fc_matrices import nw_color_map
from bokeh.io import show
scan2schedule_dict, scan2hand_dict, schedule2evonsets_dict, _ = get_paradigm_info(DATASET)
from .ets import root_sum_of_squares
from bokeh.models.formatters import DatetimeTickFormatter
formatter = DatetimeTickFormatter(minutes = '%Mmin:%Ssec')

def get_neural_event_hv_annots(schedule2evonsets_dict,SCH,alpha=0.5):
    annots = hv.VSpans((list(schedule2evonsets_dict[(SCH,'FTAP')]/2),list(schedule2evonsets_dict[(SCH,'FTAP')]/2+2)),label='FTAP').opts(color=TASK_COLORS['FTAP'], alpha=alpha, muted_alpha=0) * \
             hv.VSpans((list(schedule2evonsets_dict[(SCH,'BMOT')]/2),list(schedule2evonsets_dict[(SCH,'BMOT')]/2+2)),label='BMOT').opts(color=TASK_COLORS['BMOT'], alpha=alpha, muted_alpha=0) * \
             hv.VSpans((list(schedule2evonsets_dict[(SCH,'HOUS')]/2),list(schedule2evonsets_dict[(SCH,'HOUS')]/2+2)),label='HOUS').opts(color=TASK_COLORS['HOUS'], alpha=alpha, muted_alpha=0) * \
             hv.VSpans((list(schedule2evonsets_dict[(SCH,'READ')]/2),list(schedule2evonsets_dict[(SCH,'READ')]/2+2)),label='READ').opts(color=TASK_COLORS['READ'], alpha=alpha, muted_alpha=0) * \
             hv.VSpans((list(schedule2evonsets_dict[(SCH,'MUSI')]/2),list(schedule2evonsets_dict[(SCH,'MUSI')]/2+2)),label='MUSI').opts(color=TASK_COLORS['MUSI'], alpha=alpha, muted_alpha=0)
    return annots

def get_hrf_event_onsets_offsets(SCH,TAP_LABEL,tr_secs,hrf_thr=0.2):
    wavs = pd.DataFrame(columns=TASKS)
    task_hrf_onsets, task_hrf_offsets = {},{}
    # Load Task HRF simulated
    for task in TASKS:
        if task == 'FTAP':
            wavs[task] = np.loadtxt(osp.join(SCHEDULES_DIR,f'model_timing.{TAP_LABEL}.{SCH}.SPMG1.1D'), comments='#')
        else:
            wavs[task] = np.loadtxt(osp.join(SCHEDULES_DIR,f'model_timing.{task}.{SCH}.SPMG1.1D'), comments='#')
        # Compute reponse onset and offset
        task_hrf_onsets[task]  = [(i-1)*tr_secs for i in wavs[task][(wavs[task]>.2).astype(int).diff() == 1].index.to_list()]
        task_hrf_offsets[task] = [(i+1)*tr_secs for i in wavs[task][(wavs[task]>.2).astype(int).diff() == -1].index.to_list()]
    return task_hrf_onsets, task_hrf_offsets
    
def get_hrf_event_hv_annots(schedule2evonsets_dict,SCH,TAP_LABEL,tr_secs,hrf_thr=0.2,alpha=0.5):
    task_hrf_onsets, task_hrf_offsets = get_hrf_event_onsets_offsets(SCH,TAP_LABEL,tr_secs,hrf_thr)
    # Plot the annotations
    annots = hv.VSpans((task_hrf_onsets['FTAP'],task_hrf_offsets['FTAP']), label='FTAP').opts(color=TASK_COLORS['FTAP'], alpha=alpha, muted_alpha=0) * \
             hv.VSpans((task_hrf_onsets['BMOT'],task_hrf_offsets['BMOT']), label='BMOT').opts(color=TASK_COLORS['BMOT'], alpha=alpha, muted_alpha=0) * \
             hv.VSpans((task_hrf_onsets['HOUS'],task_hrf_offsets['HOUS']), label='HOUS').opts(color=TASK_COLORS['HOUS'], alpha=alpha, muted_alpha=0) * \
             hv.VSpans((task_hrf_onsets['READ'],task_hrf_offsets['READ']), label='READ').opts(color=TASK_COLORS['READ'], alpha=alpha, muted_alpha=0) * \
             hv.VSpans((task_hrf_onsets['MUSI'],task_hrf_offsets['MUSI']), label='MUSI').opts(color=TASK_COLORS['MUSI'], alpha=alpha, muted_alpha=0)
    return annots


def _process_roi_data(data,data_type,tr_secs,x_coord_name,y_coord_name):
    """ 
    This function help go from xarray to dataframes that are ready for heatmap plotting. This function is intended to be used
    only at the roi level [Not for edges].

    Inputs
    ======
    data: xr.DataArray with roi timeseries
    data_type: BOLD or Neural
    ts_secs: tr in seconds
    x_coord_name: name of the coordinate that should be the x axis. (e.g., tr, time, etc.)
    y_coord_name: name of the coordinate that should be the y axis. (e.g., roi, ROI, etc.)

    Returns
    =======
    data: same as input but re-sorted by network membership
    data_df: tidy version of the input data.
    nw_names: network names.
    schedule: task schedule.
    tap_label: task hand.
    x_coord_name: new name for the coordinate to be used in the X axis
    y_coord_name: new name for the coordinate to be used in the Y axis
    roi_name_2_roi_idx: dictionary to go from ROI_id to ROI_Name
    """

    # Basic checks on the input
    assert isinstance(data,xr.DataArray), "++ [cplot_roits] ERROR: data is not an instance of xr.DataArray"
    assert 'schedule' in data.attrs,      "++ [cplot_roits] ERROR: schedule information not available as an attribute"
    assert 'tap_label' in data.attrs,     "++ [cplot_roits] ERROR: tap_label information not available as an attribute"
    assert y_coord_name in data.coords, "++ [cplot_roits] ERROR: %s is not a coordinate in the input data" % y_coord_name
    assert x_coord_name in data.coords,  "++ [cplot_roits] ERROR: %s is not a coordinate in the input data" % x_coord_name

    # The data might be organized according to NW within each hemisphere. This is a problem for this code. To resolve
    # it is key to first ensure that the ROIs are sorted according to network name
    data             = data.copy()
    data_rois_idx    = pd.MultiIndex.from_tuples([tuple(r.split('_',2)) for r in data.coords[y_coord_name].values],names=['Hemisphere','Network','ROI_Name'])
    data_rois_idx, _ = data_rois_idx.sortlevel('Network')
    new_rois_order   = ['_'.join(r) for r in data_rois_idx]
    data             = data.sel(roi=new_rois_order)

    # Once data is sorted, we can now go ahead and extract network names, etc...
    nw_names         = list(data_rois_idx.get_level_values('Network').unique())
    num_nws          = len(nw_names)

    # Extract basic information
    Nrois = data.coords[y_coord_name].shape[0]
    Nacqs = data.coords[x_coord_name].shape[0]
    roi_name_2_roi_idx = {roi:i for i,roi in enumerate(data.coords[y_coord_name].values)}

    # Go from TR to secs
    if tr_secs is not None:
        new_time_coord_values = [acq * tr_secs for acq in data.coords[x_coord_name].values]
        data.coords[x_coord_name] = new_time_coord_values
        data = data.rename({x_coord_name:'Time [sec]'})
        x_coord_name = 'Time [sec]'

    # Extract schedule and tap_label
    schedule        = data.schedule
    tap_label       = data.tap_label

    # Go to tidy data format
    data_df         = data.to_dataframe()
    if 'dt' in data_df.columns:
        data_df.drop('dt',axis=1,inplace=True)
    data_df.columns = [data_type]

    # Re-index data with multi-index (Hemi, Nw, ROI)
    orig_idx      = pd.MultiIndex.from_tuples([tuple(r.split('_',2)) for r in data_df.index.get_level_values(y_coord_name)],names=['Hemisphere','Network','ROI_Name'])
    data_df       = data_df.reset_index()
    data_df.index = orig_idx

    data_df[y_coord_name] = [roi_name_2_roi_idx[i] for i in list(data_df[y_coord_name].values)]
    return data, data_df, nw_names, schedule, tap_label, x_coord_name,y_coord_name, roi_name_2_roi_idx

def _get_roits_schedule_avg(data, schedule, denoising, model=None, criteria=None, 
                            only_negatives=False, only_positives=False, 
                            rwin_dur=None, rwin_mode='mean',
                            values_cap = None):
    """
    This function will create a schedule average for ROI timseries. Its ouput can then be provided to the plotting function.

    Inputs
    ------
    data: xr.Dataset
    schedule: selected schedule
    denoising: selected denoiing method
    model: selected model (for PFM outputs only)
    criteria: selected criteria (for PFM outputs only)
    only_negatives: remove positive values prior to averaging across scans
    only_positives: remove negative values prior to averaging across scans
    rwin_dur: duration for temporal rolling window to be applied prior to averaging across scans
    rwin_mode: function to summarize data when doing rolling window [default='mean', options='mean','median','min','max'].
    values_cap: cap value to be applied prior to averaging across scans. All values greater than the cap value will be set to the cap value
    Returns
    -------
    output: xr.DataArray with the averaged data.
    """
    # Check if data is PFM input or output
    if (model is None) | (criteria is None):
        pfm_outputs = False
    else:
        pfm_outputs = True
        
    # Select dataarrays of interest from dataset provided as input
    if pfm_outputs:
        selected_dataarrays = data.filter_by_attrs(schedule=schedule, denoising=denoising)
    else:
        selected_dataarrays = data.filter_by_attrs(schedule=schedule, denoising=denoising, model=model, criteria=criteria)
    
    # Capping input data (if requested)
    if values_cap is not None:
        for da_name,da in selected_dataarrays.items():
            selected_dataarrays[da_name] = xr.where(da < -values_cap , -values_cap, da)
            selected_dataarrays[da_name] = xr.where(da > values_cap ,  values_cap, da)

    # Remove positive values (if requested) & stack in preparation for averaging
    stacked_dataarray = xr.concat([selected_dataarrays[name] for name in selected_dataarrays.data_vars], dim='scan')
    
    if only_negatives:
        stacked_dataarray = xr.concat([xr.where(da > 0, 0, da) for da in selected_dataarrays.values()], dim='scan')
    if only_positives:
        stacked_dataarray = xr.concat([xr.where(da < 0, 0, da) for da in selected_dataarrays.values()], dim='scan')
    
    stacked_dataarray.name = 'stacked'

    # Apply rolling window (if requested)
    if (rwin_dur is not None) and (rwin_dur > 1):
        if rwin_mode == 'mean':
            stacked_dataarray = stacked_dataarray.rolling(tr=rwin_dur, center=False).mean()
        if rwin_mode == 'median':
            stacked_dataarray = stacked_dataarray.rolling(tr=rwin_dur, center=False).median()
        if rwin_mode == 'min':
            stacked_dataarray = stacked_dataarray.rolling(tr=rwin_dur, center=False).min()
        if rwin_mode == 'max':
            stacked_dataarray = stacked_dataarray.rolling(tr=rwin_dur, center=False).max()
        stacked_dataarray = stacked_dataarray.fillna(0)
    
    # Average all scans part of the selected schedule
    mean_dataarray = stacked_dataarray.mean(dim='scan')
    mean_dataarray.attrs['schedule']  = schedule
    mean_dataarray.attrs['tap_label'] = 'LTAP'
    return mean_dataarray
    
def create_roits_figure(data,data_type,tr_secs,
                      x_coord_name,y_coord_name,
                      width=2000, height=500, cmap='RdBu_r', 
                      vmin=-0.02, vmax=0.02, 
                      rssts_min=None, rssts_max=None, 
                      roits_min=None, roits_max=None,
                      time_segment='All', hrf_thr=0.2):
    """
    Creates a figure with three plots: 
    (1) A carpet plot of the ROI timeseries sorted and organized by network.
    (2) Timeseries or RSS across all ROIs included in the carpet plot.
    (3) Trace of RSS across a given scan segment for all ROIs in the plot.

    Inputs:
    -------
    data: xr.DataArray with the data to be plotted.
    data_type: BOLD or Neural
    x_coord_name: name of coordinate to be used as X-axis.
    y_coord_name: name of coordinate to be used as Y-axis.
    width: total figure width
    height: total figure height
    cmap: colormap for the carpet plot
    vmin,vmax: min and max values for the carplet plot colorbar
    rssts_min, rssts_max: min and max values for the RSS timeseries
    roits_min, roits_max: min and max values for the ROI timeseries
    time_segment:
    hrf_thr: threshold for task HRF
    
    Returns:
    --------
    out_figure: hvplot figure.
    """
    # Extract basic information
    # =========================
    data, data_df, nw_names, schedule, tap_label,x_coord_name,y_coord_name,roi_name_2_roi_idx  = _process_roi_data(data,data_type,tr_secs,x_coord_name,y_coord_name)
    Nrois = data.coords[y_coord_name].shape[0]
    Nacqs = data.coords[x_coord_name].shape[0]
    
    # Prepare network related annotations
    # ===================================
    # 1. Yticks and Yticklabels
    num_nws       = len(nw_names)
    data_rois_idx = pd.MultiIndex.from_tuples([tuple(r.split('_',2)) for r in data.coords[y_coord_name].values],names=['Hemisphere','Network','ROI_Name'])
    rois_per_nw  = data_rois_idx.get_level_values('Network').value_counts().to_dict()
    rois_per_nw  = np.array([rois_per_nw[nw] for nw in nw_names])
    net_edges    = [0]+ list(rois_per_nw.cumsum())                                    # Position of horizontal dashed lines for nw delineation
    yticks       = [int(i) for i in net_edges[0:-1] + np.diff(net_edges)/2]           # Position of nw names
    yticks_info  = list(tuple(zip(yticks, nw_names)))
    # 2. Colored segments
    nw_col_segs = hv.Segments((tuple(-0.5*np.ones(num_nws)),
                                   tuple(np.array(net_edges[:-1])-0.5),
                                   tuple(-0.5*np.ones(num_nws)),
                                   tuple(np.array(net_edges[1:])-0.5), nw_names), vdims='Networks').opts(cmap=nw_color_map, color=dim('Networks'), line_width=10,show_legend=False)
    # 3. Dashed horizontal lines delineating networks
    for x in net_edges:
            nw_col_segs  = nw_col_segs  * hv.HLine(x-.5).opts(line_color='k',line_dash='dashed', line_width=0.5)

    # Prepare task-related annotations
    # ================================
    if data_type == 'BOLD':
        task_annot = get_hrf_event_hv_annots(schedule2evonsets_dict,schedule,tap_label,tr_secs,hrf_thr=0.2,alpha=0.25).opts(ylim=(0,Nrois))
    else:
        task_annot = get_neural_event_hv_annots(schedule2evonsets_dict,schedule,tap_label,alpha=0.25).opts(ylim=(0,Nrois))

    # Construct Gridspec Elements
    # ===========================
    # Carpet Plot
    # -----------
    cplot = data_df.reset_index().hvplot.heatmap(x=x_coord_name,y=y_coord_name,C=data_type, width=int(0.8*width), height=int(0.8 * height),
                                         cmap=cmap, hover_cols=['Hemisphere','Network','ROI_Name'],
                                         ylabel='ROIs [%s]' % str(Nrois), yticks=yticks_info, 
                                         clim=(vmin,vmax)).opts(xrotation=90,colorbar_opts={'title':data_type})
    cplot = cplot * nw_col_segs
    cplot = cplot * task_annot
    cplot.opts(xlim=(-0.5,data_df[x_coord_name].max()),ylim=(0,Nrois))
    # RSS TS
    # ------
    rssts = data_df.groupby(x_coord_name).agg(root_sum_of_squares)[data_type]
    rssts.name = 'RSS TS'
    rssts = pd.DataFrame(rssts).reset_index()
    # Extract min and max if not provided
    if rssts_min is None:
        rssts_min = rssts['RSS TS'].min()
    if rssts_max is None:
        rssts_max = rssts['RSS TS'].max()
    rssts = rssts.hvplot(x=x_coord_name, y='RSS TS', c='k', ylim=(rssts_min,rssts_max),width=int(0.8*width), height=int(0.4 * height)) * task_annot.opts(show_legend=False)

    # ROI TS
    # ------
    # Get onset and offsets per task
    if data_type == 'BOLD':
        task_onsets, task_offsets = get_hrf_event_onsets_offsets(schedule,tap_label,tr_secs,hrf_thr)
    else:
        task_onsets, task_offsets = {},{}
        for task in TASKS:      
            task_onsets[task]  = schedule2evonsets_dict[(this_scan_roits_xr.schedule,task)]/2
            task_offsets[task] = task_onsets[task]+2
    # Get trace for the full scan            
    roits = data_df.groupby(y_coord_name).agg(root_sum_of_squares)[data_type]
    roits.name = 'ROI TS [All]'
    roits = pd.DataFrame(roits).reset_index()
    
    # Get task specific traces 
    for task in TASKS:
        sel_acqs = list(np.concatenate([np.arange(i-1,j+1) for i,j in zip(task_onsets[task],task_offsets[task])]))
        roits['ROI TS ['+task+']'] = data_df[data_df[x_coord_name].isin(sel_acqs)].groupby(y_coord_name).agg(root_sum_of_squares)[data_type]
        
    # Extract min and max if not provided
    if roits_min is None:
        roits_min = roits['ROI TS ['+time_segment+']'].min()*0.95
    if roits_max is None:
        roits_max = roits['ROI TS ['+time_segment+']'].max()*1.05
    roi_idx_2_hemi  = {v:k.split('_')[0] for k,v in roi_name_2_roi_idx.items()}
    roi_idx_2_nw    = {v:k.split('_')[1] for k,v in roi_name_2_roi_idx.items()}
    roi_idx_2_name  = {v:k.split('_',2)[2] for k,v in roi_name_2_roi_idx.items()}
    roits['Hemisphere'] = [roi_idx_2_hemi[i] for i in roits['roi'].values]
    roits['Network']    = [roi_idx_2_nw[i] for i in roits['roi'].values]
    roits['ROI_Name']   = [roi_idx_2_name[i] for i in roits['roi'].values]
    roits = roits.hvplot.scatter(y=y_coord_name, x='ROI TS ['+time_segment+']', c='Network', cmap=nw_color_map, hover_cols=['Network','Hemisphere','ROI_Name'], xlim=(roits_min,roits_max), ylim=(0,Nrois), legend=False, height=int(0.83 * height), width=int(0.10 * width)).opts(toolbar='above')

    # Construct Final Figure
    # ======================
    #out_figure = pn.Row((cplot+rssts).cols(1),pn.Column(roits,pn.Spacer(styles=dict(background='green'))))
    out_figure = pn.Row(pn.Column(cplot,rssts),pn.Column(roits,pn.Spacer(styles=dict(background='green'))))
    return out_figure    
    
    

# ===================
# OLD STUFF TO REMOVE
# ===================
def cplot_roits(data,data_type,tr_secs,y_coord_name='roi',x_coord_name='tr', 
                vmin=None,vmax=None, cmap='RdBu_r', cplot_width=1800, cplot_height=500,
                show_nw_annot=True):
    assert isinstance(data,xr.DataArray), "++ [cplot_roits] ERROR: data is not an instance of xr.DataArray"
    assert 'schedule' in data.attrs,      "++ [cplot_roits] ERROR: schedule information not available as an attribute"
    assert 'tap_label' in data.attrs,     "++ [cplot_roits] ERROR: tap_label information not available as an attribute"
    assert y_coord_name in data.coords, "++ [cplot_roits] ERROR: %s is not a coordinate in the input data" % y_coord_name
    assert x_coord_name in data.coords,  "++ [cplot_roits] ERROR: %s is not a coordinate in the input data" % x_coord_name
    data = data.copy()
    # Extract basic information
    Nrois = data.coords[y_coord_name].shape[0]
    Nacqs = data.coords[x_coord_name].shape[0]
    roi_name_2_roi_idx = {roi:i for i,roi in enumerate(data.coords[y_coord_name].values)}
    # Go from TR to secs
    if tr_secs is not None:
        new_time_coord_values = [pd.to_timedelta(acq * tr_secs, unit='s') for acq in data.coords[x_coord_name].values]
        data.coords[x_coord_name] = new_time_coord_values
        data = data.rename({x_coord_name:'Time'})
        x_coord_name = 'Time'
        
    schedule  = data.schedule
    tap_label = data.tap_label
    data_df = data.to_dataframe()
    if 'dt' in data_df.columns:
        data_df.drop('dt',axis=1,inplace=True)
    data_df.columns = [data_type]

    # Re-index data with multi-index (Hemi, Nw, ROI)
    orig_idx = pd.MultiIndex.from_tuples([tuple(r.split('_',2)) for r in data_df.index.get_level_values(y_coord_name)],names=['Hemisphere','Network','ROI_Name'])
    data_df = data_df.reset_index()
    data_df.index = orig_idx

    # Extract min and max if not provided
    if vmin is None:
        vmin = data_df[data_type].quantile(0.025)
    if vmax is None:
        vmax = data_df[data_type].quantile(0.0975)

    data_df['roi'] = [roi_name_2_roi_idx[i] for i in list(data_df['roi'].values)]

    # Extract network names
    if show_nw_annot:
        nw_names = list(data_df.index.get_level_values('Network').unique())
        num_nws            = len(nw_names)
        # Annotation positions
        rois_per_nw  = orig_idx.get_level_values('Network').value_counts().to_dict()
        rois_per_nw  = np.array([rois_per_nw[nw] for nw in nw_names])
        print(rois_per_nw)
        net_edges    = [0]+ list(rois_per_nw.cumsum())                                    # Position of horizontal dashed lines for nw delineation
        yticks       = [int(i) for i in net_edges[0:-1] + np.diff(net_edges)/2]           # Position of nw names
        yticks_info  = list(tuple(zip(yticks, nw_names)))
        print(yticks_info)
        # Annotation 1 - Nw Colors
        nw_col_segs = hv.Segments((tuple(-.5*np.ones(num_nws)),tuple(np.array(net_edges[:-1])-0.5),
                               tuple(-.5*np.ones(num_nws)),tuple(np.array(net_edges[1:])-0.5), nw_names), vdims='Networks').opts(cmap=nw_color_map, color=dim('Networks'), line_width=10,show_legend=False)
                
        output_plot = data_df.reset_index().hvplot.heatmap(x=x_coord_name,y=y_coord_name,C=data_type, 
                                         xformatter=formatter, width=cplot_width, height=cplot_height, cmap=cmap, hover_cols=['Hemisphere','Network','ROI_Name'],
                                         xlabel='Time', ylabel='ROIs [%s]' % str(Nrois), yticks=yticks_info, 
                                         clim=(vmin,vmax)).opts(xrotation=90,colorbar_opts={'title':data_type}) 
        for x in net_edges:
            output_plot = output_plot * hv.HLine(x-.5).opts(line_color='k',line_dash='dashed', line_width=0.5) 
        #output_plot = output_plot * nw_col_segs
        
    else:
        output_plot = data_df.hvplot.heatmap(x=x_coord_name,y=y_coord_name,C=data_type, 
                                             xformatter=formatter, width=cplot_width, height=cplot_height, cmap=cmap,  
                                             clim=(vmin,vmax)).opts(xrotation=90,colorbar_opts={'title':data_type})
    return output_plot
    
    
def carpet_plot_roits(data, roi_info, scan_name, data_type, nw_names=None, 
                          show_title=True, cmap='RdBu_r', vmin=None, vmax=None,cbar_title='Unknown', 
                          only_negatives=False, only_positives=False, rolling_win_dur=0, rolling_win_func='mean', 
                          show_time_rss=False, show_roi_rss=False, 
                          show_neural_event_annots=False, show_hrf_event_annots=False,hrf_thr=0.2, roi_rss_period='Whole', only_positives_rssts=False, only_positives_rssroi=False):
    """ Generate Carpet Plot for ROI timeseries

    INPUTS:
    -------
    data: an xr.Dataset with one or more xr.Dataarrays
    roi_info: a pd.Dataframe with the following columns ['ROI_ID', 'Hemisphere', 'Network', 'ROI_Name', 'pos_R', 'pos_A', 'pos_S', 'color_R', 'color_G', 'color_B', 'RGB']
    show_neural_event_annots: show task segments with neural timing [default=False] 
    show_hrf_event_annots: show task segments with hrf timing [default=False]
    hrf_thr: threshold for deciding onset and offset of HRF-based timing [default=0.2]
    """
    # Obtain info
    this_scan_roits_xr, out = None, None
    if scan_name is None:
        this_scan_roits_xr = data.sel(dt=data_type)
    else:
        this_scan_roits_xr = data[scan_name].sel(dt=data_type)
    
    Nacqs,Nrois         = this_scan_roits_xr.shape
    # Create ROI Multi-index
    # ======================
    orig_idx           = pd.MultiIndex.from_tuples([s.split('_',2) for s in  data.roi.values], names=['Hemisphere','Network','ROI_Name'])
    this_scan_roits_df = pd.DataFrame(this_scan_roits_xr.values, columns=orig_idx).T
    
    # Modify data only for plotting purposes (if requested)
    # =====================================================
    if only_negatives:
        this_scan_roits_df[this_scan_roits_df>0] = np.nan
    if only_positives:
        this_scan_roits_df[this_scan_roits_df<0] = np.nan
    if rolling_win_dur >0:
        if rolling_win_func == 'mean':
            this_scan_roits_df = this_scan_roits_df.rolling(rolling_win_dur, center=True).mean()
        if rolling_win_func == 'min':
            this_scan_roits_df = this_scan_roits_df.rolling(rolling_win_dur, center=True).min()
        if rolling_win_func == 'max':
            this_scan_roits_df = this_scan_roits_df.rolling(rolling_win_dur, center=True).max()
        if rolling_win_func == 'median':
            this_scan_roits_df = this_scan_roits_df.rolling(rolling_win_dur, center=True).median()
        if rolling_win_func == 'count':
            this_scan_roits_df = this_scan_roits_df.rolling(rolling_win_dur, center=True).count()
    if vmin is None:
        vmin = np.quantile(this_scan_roits_df.values.flatten(),0.025)
    if vmax is None:
        vmax = np.quantile(this_scan_roits_df.values.flatten(),0.975)
        
    # Extract network names from the passed dataarrays
    # =================================================
    this_scan_roits_df = this_scan_roits_df.sort_values(by=['Network','Hemisphere'])
    if nw_names is None:
        nw_names = this_scan_roits_df.index.get_level_values('Network').unique()
    num_nws            = len(nw_names)
    # Annotation positions
    # ====================
    rois_per_nw  = rois_per_nw  = this_scan_roits_df.groupby('Network').size().values 
    net_edges    = [0]+ list(rois_per_nw.cumsum())                                    # Position of horizontal dashed lines for nw delineation
    yticks       = [int(i) for i in net_edges[0:-1] + np.diff(net_edges)/2]           # Position of nw names
    yticks_info  = list(tuple(zip(yticks, nw_names)))
    
    # Annotation 1 - Nw Colors
    nw_col_segs = hv.Segments((tuple(-.5*np.ones(num_nws)),tuple(np.array(net_edges[:-1])-0.5),
                               tuple(-.5*np.ones(num_nws)),tuple(np.array(net_edges[1:])-0.5), nw_names), vdims='Networks').opts(cmap=nw_color_map, color=dim('Networks'), line_width=10,show_legend=False)
    
    # Create heatmap
    aux = this_scan_roits_df.reset_index(drop=True)
    aux.index.name = 'ROIs'
    aux.columns.name = 'Time'
    #scan_heatmap = aux.hvplot.heatmap(width=1800, height=500, cmap=cmap, yticks=yticks_info, xlabel='Time', ylabel='ROIs [%s]' % str(Nrois), clim=(vmin,vmax)).opts(xrotation=90,colorbar_opts={'title':cbar_title}) #, colorbar_position='top')
    scan_heatmap = aux.hvplot.heatmap(width=1800, height=500, cmap=cmap, yticks=yticks_info, xlabel='Time', ylabel='ROIs [%s]' % str(Nrois), clim=(vmin,vmax)).opts(xrotation=90,colorbar=False) #, colorbar_position='top')
    
    # Annotation 2 - Nw Horizontal Lines
    for x in net_edges:
        scan_heatmap = scan_heatmap * hv.HLine(x-.5).opts(line_color='k',line_dash='dashed', line_width=0.5)
    
    # Annotation 3 - Event colorbands
    if show_neural_event_annots or show_hrf_event_annots:
        this_scan_schedule  = this_scan_roits_xr.schedule
        this_scan_tap_label = this_scan_roits_xr.tap_label
    if show_neural_event_annots:
        scan_heatmap_annot = scan_heatmap * get_neural_event_hv_annots(schedule2evonsets_dict,this_scan_schedule,0.25) * nw_col_segs
    elif show_hrf_event_annots:
        scan_heatmap_annot = scan_heatmap * get_hrf_event_hv_annots(schedule2evonsets_dict,this_scan_schedule,this_scan_tap_label,hrf_thr=hrf_thr,alpha=0.25).opts(ylim=(0,Nrois)) * nw_col_segs
    else:
        scan_heatmap_annot = scan_heatmap * nw_col_segs
    
    scan_heatmap_annot.opts(ylim=(0,Nrois), xlim=(-1,Nacqs-1),legend_position='top_left')
    
    if (show_time_rss==True):
        aux = this_scan_roits_df.copy()
        if only_positives_rssts:
            aux[aux<=0] = np.nan
        rssts            = np.sqrt(np.sum(aux.reset_index(drop=True).T**2, axis=1))
        rssts.index.name ='Time'
        rssts.columns    = ['Time RSS']
        rssts.name       = 'Time RSS'
        rssts_plot       = rssts.hvplot(color='k', width=1800, height=200, ylabel='RSST')
        if show_neural_event_annots:
            rssts_plot = rssts_plot * get_neural_event_hv_annots(schedule2evonsets_dict,this_scan_schedule,0.25)
            rssts_plot.opts(legend_position='bottom')
        if show_hrf_event_annots:
            rssts_plot = rssts_plot * get_hrf_event_hv_annots(schedule2evonsets_dict,this_scan_schedule,this_scan_tap_label,hrf_thr=hrf_thr,alpha=0.25)
            rssts_plot.opts(legend_position='bottom')
        
    if show_roi_rss==True:
        if show_hrf_event_annots:
            task_onsets, task_offsets = get_hrf_event_onsets_offsets(this_scan_roits_xr.schedule,this_scan_roits_xr.tap_label,hrf_thr)
        if show_neural_event_annots:
            task_onsets, task_offsets = {},{}
            for task in TASKS:      
                task_onsets[task]  = schedule2evonsets_dict[(this_scan_roits_xr.schedule,task)]/2
                task_offsets[task] = task_onsets[task]+2
        # Compute ROI RSS across full scan
        aux = this_scan_roits_df.copy()
        if only_positives_rssroi:
            aux[aux<=0] = np.nan
        rssroi            = pd.DataFrame(np.sqrt(np.sum(aux.reset_index(drop=True)**2,axis=1)).values, columns=['Whole'])
        rssroi.index = this_scan_roits_df.index
        rssroi.reset_index(inplace=True)
        rssroi.index.name ='ROIs'
        # Compute ROI RSS across task-specific segments
        # =============================================
        for task in TASKS:
            sel_acqs = list(np.concatenate([np.arange(i-1,j+1) for i,j in zip(task_onsets[task],task_offsets[task])]))
            rssroi[task] = np.sqrt(np.sum(aux[sel_acqs].reset_index(drop=True)**2,axis=1)).values
        #rssroi.index = this_scan_roits_df.index
        rssroi['ROI_ID'] = ['_'.join(t) for t in this_scan_roits_df.index.values] #list(this_scan_roits_df.index.get_level_values('ROI_Name'))
        rssroi.name       = 'ROI RSS'
        
        min_x, max_x = rssroi[roi_rss_period].quantile(0.0275),rssroi[roi_rss_period].quantile(0.975)
        rssroi_plot = rssroi.hvplot.scatter(x=roi_rss_period, y='ROIs', color='Network', width=200, height=500, ylim=(0,Nrois-1), xlim=(min_x,max_x), hover_cols=['Hemisphere','Network','ROI_Name'],s=5, legend=False, cmap=nw_color_map, ylabel='RSS for %s ROIs' % str(Nrois))
        for x in net_edges:
            rssroi_plot = rssroi_plot * hv.HLine(x-.5).opts(line_color='k',line_dash='dashed', line_width=0.5)
    
    # Final layout formation
    # ======================
    if (show_time_rss==True) and (show_roi_rss==False):
        return pn.Column(scan_heatmap_annot,rssts_plot)
    elif (show_roi_rss==True) and (show_time_rss==False):
        return pn.Row(scan_heatmap_annot,rssroi_plot)
    elif (show_roi_rss==True) and (show_time_rss==True):
        return pn.Row(pn.Column(scan_heatmap_annot,rssts_plot),rssroi_plot)
    else:
        return pn.Row(scan_heatmap_annot)
    return out

def carpet_hvplot_ets_oneNW(data, roi_info, scan_name, data_type, sel_network, nw_names=None, 
                            show_title=True, cmap='RdBu_r', vmin=None, vmax=None,cbar_title='Unknown', 
                            only_negatives=False, only_positives=False, rolling_win=None, 
                            show_time_rss=False, show_edge_rss=False, 
                            show_neural_event_annots=False, show_hrf_event_annots=False,hrf_thr=0.2, edge_rss_period='Whole',rssts_plot_ylim=None):
    """ Generate Carpet Plot for ROI timeseries

    INPUTS:
    -------
    data: an xr.Dataset with one or more xr.Dataarrays
    roi_info: a pd.Dataframe with the following columns ['ROI_ID', 'Hemisphere', 'Network', 'ROI_Name', 'pos_R', 'pos_A', 'pos_S', 'color_R', 'color_G', 'color_B', 'RGB']
    show_neural_event_annots: show task segments with neural timing [default=False] 
    show_hrf_event_annots: show task segments with hrf timing [default=False]
    hrf_thr: threshold for deciding onset and offset of HRF-based timing [default=0.2]
    """
    # Select edges
    if scan_name is None:
        edge_selection   = [e for e in list(data.edge.values) if (sel_network in e) ]
    else:
        edge_selection   = [e for e in list(data[scan_name].edge.values) if (sel_network in e) ]
    
    data = data.sel(edge=edge_selection)
    
    # Obtain info
    if scan_name is None:
        this_scan_ets_xr = data.sel(dt=data_type)
    else:
        this_scan_ets_xr = data[scan_name].sel(dt=data_type)
    Nacqs,Nedges           = this_scan_ets_xr.shape
    
    # Move data to a dataframe
    # ========================
    orig_edge_index  = pd.MultiIndex.from_tuples([tuple(e.split('-')[0].split('_',2)+e.split('-')[1].split('_',2)) for e in data.edge.values],names=['ROIx_Hemisphere','ROIx_Network','ROIx_ROI','ROIy_Hemisphere','ROIy_Network','ROIy_ROI'])
    this_scan_ets_df = pd.DataFrame(this_scan_ets_xr.values, columns=orig_edge_index).T
    
    # Transform data (if needed)
    # ==========================
    if only_negatives:
        this_scan_ets_df[this_scan_ets_df>0] = np.nan
    if only_positives:
        this_scan_ets_df[this_scan_ets_df<0] = np.nan
    if rolling_win is not None:
        this_scan_ets_df = this_scan_ets_df.rolling(rolling_win, center=True).min()
    # Extract min, max values automatically if user provided none
    # ===========================================================
    if vmin is None:
        vmin = np.quantile(this_scan_ets_df.values.flatten(),0.025)
    if vmax is None:
        vmax = np.quantile(this_scan_ets_df.values.flatten(),0.975)
    
    # Get index per network & sort them
    # =================================
    nw_names    = list(roi_info['Network'].unique())
    num_nws     = len(nw_names)
    aux         = this_scan_ets_df.copy()
    per_nw_idxs = {}
    final_edge_index = []
    for nw in nw_names:
        if nw != sel_network:
            per_nw_idxs[nw] = [i for i in aux.index if nw in i]
        else:
            per_nw_idxs[nw] = [i for i in aux.index if ((i[1]==sel_network) and (i[4]==sel_network))]
        aux.drop(per_nw_idxs[nw],inplace=True)
        final_edge_index = final_edge_index + per_nw_idxs[nw]    
    this_scan_ets_df     = this_scan_ets_df.loc[final_edge_index]
    
    # Get Y-tick location per nw and dashed lines
    # ===========================================
    edges_per_nw = np.array([len(per_nw_idxs[nw]) for nw in nw_names])  # Number of ROIs per network
    net_limits   = [0]+ list(edges_per_nw.cumsum())                                    # Position of horizontal dashed lines for nw delineation
    yticks       = [int(i) for i in net_limits[0:-1] + np.diff(net_limits)/2]           # Position of nw names
    yticks_info  = list(tuple(zip(yticks, nw_names)))
    
    # Create Network colored segments
    # ===============================
    nw_col_segs = hv.Segments((tuple(-.5*np.ones(num_nws)),tuple(np.array(net_limits[:-1])-0.5),
                               tuple(-.5*np.ones(num_nws)),tuple(np.array(net_limits[1:])-0.5), nw_names), vdims='Networks').opts(cmap=nw_color_map, color=dim('Networks'), line_width=10,show_legend=False)
    
    # Create heatmap
    aux = this_scan_ets_df.copy()
    aux.index = np.arange(aux.shape[0])
    aux.index.name = 'Edges'
    aux.columns.name = 'Time'
    scan_heatmap = aux.hvplot.heatmap(width=1800, height=500, cmap=cmap, xlabel='Time', ylabel='Edges from %s' % sel_network, clim=(vmin,vmax), yticks=yticks_info,ylim=(0,Nedges), xlim=(-1,Nacqs-1)).opts(xrotation=90,colorbar_opts={'title':cbar_title}) #, colorbar_position='top')
    # Add horizontal dashed lines for nw
    # ==================================
    for x in net_limits:
        scan_heatmap = scan_heatmap * hv.HLine(x-.5).opts(line_color='k',line_dash='dashed', line_width=0.5, active_tools=[])
    
    # Task Event Annotations
    # ======================
    if show_neural_event_annots or show_hrf_event_annots:
        this_scan_schedule  = this_scan_ets_xr.schedule
        this_scan_tap_label = this_scan_ets_xr.tap_label
    if show_neural_event_annots:
        scan_heatmap_annot = scan_heatmap * get_neural_event_hv_annots(schedule2evonsets_dict,this_scan_schedule,0.25) * nw_col_segs
    elif show_hrf_event_annots:
        scan_heatmap_annot = scan_heatmap * get_hrf_event_hv_annots(schedule2evonsets_dict,this_scan_schedule,this_scan_tap_label,hrf_thr=hrf_thr,alpha=0.25) * nw_col_segs
    else:
        scan_heatmap_annot = scan_heatmap * nw_col_segs
    
    scan_heatmap_annot.opts(legend_position='top_left')
    # Make VSpan dissapear on click
    scan_heatmap_annot_fig = hv.render(scan_heatmap_annot)
    scan_heatmap_annot_fig.legend.click_policy="hide"
    
    # RSS Timeseries Across ROIs
    # ==========================
    if (show_time_rss==True):
        rssts            = np.sqrt(np.sum(this_scan_ets_df.reset_index(drop=True).T**2, axis=1))
        rssts.index.name ='Time'
        rssts.columns    = ['Time RSS']
        rssts.name       = 'Time RSS'
        if rssts_plot_ylim is None:
            rssts_plot       = rssts.hvplot(color='k', width=1800, height=200, ylabel='RSST')
        else:
            rssts_plot       = rssts.hvplot(color='k', width=1800, height=200, ylabel='RSST', ylim=rssts_plot_ylim)
        if show_neural_event_annots:
            rssts_plot = rssts_plot * get_neural_event_hv_annots(schedule2evonsets_dict,this_scan_schedule,0.25)
        if show_hrf_event_annots:
            rssts_plot = rssts_plot * get_hrf_event_hv_annots(schedule2evonsets_dict,this_scan_schedule,this_scan_tap_label,hrf_thr=hrf_thr,alpha=0.25)
        rssts_plot.opts(legend_position='bottom')
        rssts_fig        = hv.render(rssts_plot)
        rssts_fig.legend.click_policy="hide"
    
    # RSS Across Time for ROIs
    # ========================
    if show_edge_rss==True:
        if show_hrf_event_annots:
            task_onsets, task_offsets = get_hrf_event_onsets_offsets(this_scan_ets_xr.schedule,this_scan_ets_xr.tap_label,hrf_thr)
        if show_neural_event_annots:
            task_onsets, task_offsets = {},{}
            for task in TASKS:      
                task_onsets[task]  = schedule2evonsets_dict[(this_scan_ets_xr.schedule,task)]/2
                task_offsets[task] = task_onsets[task]+2
        # Compute Edge RSS across full scan
        rssedge            = pd.DataFrame(np.sqrt(np.sum(this_scan_ets_df.reset_index(drop=True)**2,axis=1)).values, columns=['Whole'])
        rssedge.index.name ='Edges'
        # Compute Edge RSS across task-specific segments
        for task in TASKS:
            sel_acqs = list(np.concatenate([np.arange(i-1,j+1) for i,j in zip(task_onsets[task],task_offsets[task])]))
            rssedge[task] = np.sqrt(np.sum(this_scan_ets_df[sel_acqs].reset_index(drop=True)**2,axis=1)).values
        rssedge['Edge_Name'] = [('-').join([('_').join(i[0:3]),('_').join(i[3:6])]) for i in this_scan_ets_df.index]
        rssedge.name         = 'Edge RSS'
        
        min_x, max_x = rssedge[edge_rss_period].quantile(0.0275),rssedge[edge_rss_period].quantile(0.975)
        rssedge_plot        = rssedge.hvplot(x=edge_rss_period, y='Edges',color='k', width=200, height=500, ylabel='RSST', ylim=(0,Nedges-1), xlim=(min_x,max_x), hover_cols=['Edge_Name']).opts(line_width=1, xrotation=90)
        for i,nw in enumerate(nw_names):
            rssedge_plot = rssedge_plot * hv.HSpan(net_limits[i],net_limits[i+1]).opts(color = nw_color_map[nw])
        rssedge_fig        = hv.render(rssedge_plot)
    # Final layout formation
    # ======================
    if (show_time_rss==True) and (show_edge_rss==False):
        return pn.Column(scan_heatmap_annot_fig,rssts_fig)
    elif (show_edge_rss==True) and (show_time_rss==False):
        return pn.Row(scan_heatmap_annot_fig,rssedge_fig)
    elif (show_edge_rss==True) and (show_time_rss==True):
        return pn.Row(pn.Column(scan_heatmap_annot_fig,rssts_fig),rssedge_fig )# pn.Column(pn.Row(scan_heatmap_annot_fig,rssroi_fig), pn.Row(rssts_fig,None))
    else:
        return scan_heatmap_annot_fig

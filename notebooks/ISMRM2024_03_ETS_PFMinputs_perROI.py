# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: me-ets (bcbl-2024a)
#     language: python
#     name: me-ets_2024a
# ---

# # Description
#
# This notebook shows ETS carpet plots for inputs to PFM both at the single-scan and group level

# + editable=true slideshow={"slide_type": ""}
import sys
sys.path.append('../python/')

from utils.basics import PRCS_DATA_DIR, SCHEDULES_DIR, DATASET, get_paradigm_info, get_roi_info, TASKS
from utils.basics import ATLAS_DIR_17NETWORKS as ATLAS_DIR
from utils.basics import ATLAS_NAME_17NETWORKS as ATLAS_NAME

from utils.plotting import create_roits_figure, _get_roits_schedule_avg
from utils.plotting import create_ets_figure

import os.path as osp
import xarray as xr
import numpy as np

import pandas as pd
from tqdm import tqdm
import panel as pn
pn.extension()

from nilearn.maskers import NiftiLabelsMasker


# allows visualisation in notebook
from bokeh.io import output_notebook
from bokeh.resources import INLINE
output_notebook(INLINE)
# -

# # 1. Load ROI Info

roi_info, roi_names = get_roi_info(ATLAS_DIR, ATLAS_NAME)

# # 2. Load Paradigm Info

scan2schedule_dict, scan2hand_dict, schedule2evonsets_dict, _ = get_paradigm_info(DATASET)

# # 3. Load Per-task Top ROIs (according to NeuroSynth)

atlas_path = osp.join(ATLAS_DIR,f'{ATLAS_NAME}.nii.gz')
nlm = NiftiLabelsMasker(atlas_path )

for task in TASKS:
    group_spm_path = osp.join(ATLAS_DIR,f'ALL.ttest.mACF.{task}.DN.resampled.nii.gz')
    aux = nlm.fit_transform(group_spm_path)
    roi_info[task] = aux.flatten()

# + editable=true slideshow={"slide_type": ""}
roi_info.head(5)
# -

roi_select_dict = {'All': list(roi_info['ROI_Name'].values),
                   'FTAP':list(roi_info.sort_values(by='FTAP', ascending=False)[0:50]['ROI_Name'].values),
                   'HOUS':list(roi_info.sort_values(by='HOUS', ascending=False)[0:50]['ROI_Name'].values),
                   'BMOT':list(roi_info.sort_values(by='BMOT', ascending=False)[0:50]['ROI_Name'].values),
                   'READ':list(roi_info.sort_values(by='READ', ascending=False)[0:50]['ROI_Name'].values),
                   'MUSI':list(roi_info.sort_values(by='MUSI', ascending=False)[0:50]['ROI_Name'].values)}

roi_sorting_select = pn.widgets.Select(name='ROI Sorting Method', value='All',options=list(roi_select_dict.keys()), width=125)
#roi_list_multiselect = pn.widgets.MultiSelect(value=roi_select_dict['All'][0:1],options=roi_select_dict['All'],width=225,name='ROIs',size=5 )
roi_list_multiselect = pn.widgets.Select(options=roi_select_dict['All'],width=225,name='ROIs')
@pn.depends(roi_sorting_select.param.value, watch=True)
def _update_roi_list(roi_sort_mode):
    rois = roi_select_dict[roi_sort_mode]
    roi_list_multiselect.options = rois
    roi_list_multiselect.value = rois[0:1]
roi_sel_card = pn.Card(roi_sorting_select, roi_list_multiselect,title='ROI Selection', width=250)

# # 4. Load ROI Timeseries
#
# > NOTE: So we can compare ETS to originating ROI Timeseries

# + editable=true slideshow={"slide_type": ""}
roi_ts_xrs = xr.Dataset()
for sbj in tqdm(DATASET.keys(),desc='Subject'):
    for run in DATASET[sbj]:
            data_path = osp.join(PRCS_DATA_DIR,sbj,'D04_ROIts',f'{sbj}_{run}_{ATLAS_NAME}.roits.nc')
            data = xr.open_dataarray(data_path) #.sel(tr=np.arange(0,210))
            # Basic Denoising
            roi_ts_xrs[(sbj,run,'Basic')] = data.sel(dt=['E01','E02','E03'])
            roi_ts_xrs[(sbj,run,'Basic')].attrs['schedule'] = scan2schedule_dict[sbj,run]
            roi_ts_xrs[(sbj,run,'Basic')].attrs['tap_label'] = scan2hand_dict[(sbj,run)]
            roi_ts_xrs[(sbj,run,'Basic')].attrs['denoising'] = 'Basic'
            # MEICA Denoising
            aux = data.sel(dt=['E01-MEICA','E02-MEICA','E03-MEICA'])
            aux = aux.assign_coords({'dt':['E01','E02','E03']})                
            roi_ts_xrs[(sbj,run,'MEICA')] = aux
            roi_ts_xrs[(sbj,run,'MEICA')].attrs['schedule']  = scan2schedule_dict[sbj,run]
            roi_ts_xrs[(sbj,run,'MEICA')].attrs['tap_label'] = scan2hand_dict[(sbj,run)]
            roi_ts_xrs[(sbj,run,'MEICA')].attrs['denoising']  = 'MEICA'
# -

# ***
# # 5. Load ETS-level Data
#
# ## 5.1. ETS for input data

ets_xrs = xr.Dataset()
for sbj in DATASET.keys():
    for run in tqdm(DATASET[sbj], desc=sbj):
        for zscoring in ['Yes','No']:
            zscoring_lower_cease = zscoring.lower()
            file_path = osp.join(PRCS_DATA_DIR,sbj,'D05_ets',f'{sbj}_{run}_{ATLAS_NAME}.ets.{zscoring_lower_cease}Zscore.nc')
            aux       = xr.open_dataarray(file_path) #.sel(tr=np.arange(0,210))
            # Save Basic Pre-processing ones
            ets_xrs[(sbj,run,'Basic',zscoring)] = aux.sel(dt=['E01','E02','E03'])
            ets_xrs[(sbj,run,'Basic',zscoring)].attrs['schedule'] = scan2schedule_dict[sbj,run]
            ets_xrs[(sbj,run,'Basic',zscoring)].attrs['tap_label'] = scan2hand_dict[(sbj,run)]
            ets_xrs[(sbj,run,'Basic',zscoring)].attrs['denoising'] = 'Basic'
            ets_xrs[(sbj,run,'Basic',zscoring)].attrs['zscoring']  = zscoring
            # Save MEICA Pre-processing ones
            aux = aux.sel(dt=['E01-MEICA','E02-MEICA','E03-MEICA'])
            aux = aux.assign_coords({'dt':['E01','E02','E03']})
            ets_xrs[(sbj,run,'MEICA',zscoring)] = aux
            ets_xrs[(sbj,run,'MEICA',zscoring)].attrs['schedule']  = scan2schedule_dict[sbj,run]
            ets_xrs[(sbj,run,'MEICA',zscoring)].attrs['tap_label'] = scan2hand_dict[(sbj,run)]
            ets_xrs[(sbj,run,'MEICA',zscoring)].attrs['denoising'] = 'MEICA'
            ets_xrs[(sbj,run,'MEICA',zscoring)].attrs['zscoring']  = zscoring

# + [markdown] editable=true slideshow={"slide_type": ""}
# # 6. Plot ETS at the scan level for PFM inputs

# +
# Data Selection
sbj_run_select   = pn.widgets.NestedSelect(options=DATASET, layout=pn.Row, levels=[{'name':'Subject','width':100},{'name':'Run','width':100}])
echo_select      = pn.widgets.Select(name='Echo',options=['E01','E02','E03'],value='E02', width=100)
denoising_select = pn.widgets.Select(name='Denoising',options=['MEICA','Basic'], width=100)
zscoring_select  = pn.widgets.Select(name='ETS Z-scrore',options=['Yes','No'], width=100)
scan_card        = pn.Card(sbj_run_select,pn.Row(denoising_select,zscoring_select),echo_select, title='Data Selection', width=250)

# Carpet Plot Configuration
only_positives_check = pn.widgets.Checkbox(name='Show only positive values')
only_negatives_check = pn.widgets.Checkbox(name='Show only negative values')
cplot_config_card    = pn.Card(only_positives_check,only_negatives_check,title='CPlot Configuration', width=250)

tseg_select         = pn.widgets.Select(name='Scan Segment Selection',options=['All'] + TASKS, width=200, description='Select the scan segment you want to be used for computing the ROI-RS scatter on the right')
rssts_config_card   = pn.Card(tseg_select, title='ROI-TS Configuration', width=250)

control_column = pn.Column(pn.layout.Divider(), scan_card, pn.layout.Divider(), roi_sel_card, pn.layout.Divider(), cplot_config_card,pn.layout.Divider(),rssts_config_card,pn.layout.Divider() )


# + editable=true slideshow={"slide_type": ""}
@pn.depends(sbj_run_select, denoising_select, echo_select, zscoring_select, 
            only_positives_check, only_negatives_check,
            tseg_select, roi_list_multiselect)
def options_to_plot(sbj_run, denoising, echo, zscoring, only_positives, only_negatives, time_segment, seed_roi):
    assert isinstance(seed_roi,str), "++ ERROR[options_to_plot]: seed_rois is not a string"
    sbj = sbj_run['Subject']
    run = sbj_run['Run']
    scan_name = (sbj,run,denoising,zscoring)
    # Edge Selection
    all_edges = [tuple(e.split('-')) for e in list(ets_xrs[scan_name].edge.values)]
    edge_selection = ['-'.join(e) for e in all_edges if seed_roi==e[0] or seed_roi==e[1]]
    data = ets_xrs[scan_name].sel(dt=echo,edge=edge_selection)
    
    # Remove positive or regative values
    aux_attrs = data.attrs
    if only_positives:
        data = xr.where(data > 0, data, np.nan)
    if only_negatives:
        data = xr.where(data < 0, data, np.nan)
    data.attrs = aux_attrs

    # Get seed ROI timeseries
    roi_ts = roi_ts_xrs[(sbj,run,denoising)].sel(dt=echo,roi=seed_roi).to_dataframe()
    roi_ts.drop(['roi','dt'],axis=1, inplace=True)
    roi_ts.columns=['BOLD']
    roi_ts.reset_index(drop=True,inplace=True)
    roi_ts['Time [sec]'] = 1.5 * np.arange(roi_ts.shape[0])
    out = create_ets_figure(data,'BOLD',1.5,'tr','edge',seed_roi,echo,roi_ts,
                      width=2000, height=500, cmap='RdBu_r', 
                      vmin=-5, vmax=5, 
                      rssts_min=None, rssts_max=None, 
                      roits_min=None, roits_max=None,
                      time_segment=time_segment, hrf_thr=0.2)
    return out


# + editable=true slideshow={"slide_type": ""}
dashboard1 = pn.Row(control_column,options_to_plot)
dashboard1.show()
# -

# # 7. Plot ETS at the Schedule Level (Inputs to PFM)

# + editable=true slideshow={"slide_type": ""}

# -







# + editable=true slideshow={"slide_type": ""}

# -











def get_avg_per_schedule(data, sel_schedule, sel_denoising, sel_zscoring, sel_model=None,sel_criteria=None, only_negatives=False,rolling_win=None, neg_val_cap = None):
    if (sel_model is None) | (sel_criteria is None):
        selected_dataarrays = data.filter_by_attrs(schedule=sel_schedule, denoising=sel_denoising, zscoring=sel_zscoring)
    else:
        selected_dataarrays = data.filter_by_attrs(schedule=sel_schedule, denoising=sel_denoising, model=sel_model, criteria=sel_criteria, zscoring=sel_zscoring)
    if neg_val_cap is not None:
        for da_name,da in selected_dataarrays.items():
            selected_dataarrays[da_name] = xr.where(da < neg_val_cap , neg_val_cap, da)
            
    if only_negatives:
        stacked_dataarray = xr.concat([xr.where(da > 0, 0, da) for da in selected_dataarrays.values()], dim='scan')
    else:
        stacked_dataarray = xr.concat([selected_dataarrays[name] for name in selected_dataarrays.data_vars], dim='scan')

    stacked_dataarray.name = 'stacked'
    if rolling_win is not None:
        stacked_dataarray = stacked_dataarray.rolling(tr=rolling_win, center=False).mean()
        stacked_dataarray = stacked_dataarray.fillna(0)
    mean_dataarray = stacked_dataarray.mean(dim='scan')
    mean_dataarray.attrs['schedule']  = sel_schedule
    mean_dataarray.attrs['tap_label'] = 'LTAP'
    return mean_dataarray


# + editable=true slideshow={"slide_type": ""}
@pn.depends(sbj_run_select,denoising_select,zscoring_select,tseg_select, roi_list_multiselect, cplot_posonly_check, cplot_negonly_check, cplot_rwin_dur_select, cplot_rwin_mode_select)
def get_edge_carpetplot_hrf_scan(sbj_run, denoising,zscoring,tseg,roi_list,cplot_posonly_val,cplot_negonly_val, cplot_rwin_dur_val,cplot_rwin_mode_val):
    sbj = sbj_run['Subject']
    run = sbj_run['Run']
    return carpet_hvplot_ets_ROIlist(ets_xrs, roi_info,(sbj,run,denoising,zscoring), 'E02', roi_list, vmin=-5, vmax=5, 
                                     show_hrf_event_annots=True, show_time_rss=True, show_edge_rss=True, 
                                     edge_rss_period=tseg, 
                                     only_positives=cplot_posonly_val, only_negatives=cplot_negonly_val, 
                                     rolling_win=cplot_rwin_dur_val,rolling_mode=cplot_rwin_mode_val, 
                                     cmap='RdBu_r', cbar_title='E02 | Scan Level')


# -

from utils.plotting import get_hrf_event_hv_annots, get_hrf_event_onsets_offsets, get_neural_event_hv_annots
def carpet_hvplot_ets_ROIlist(data, roi_info, scan_name, data_type, roi_list, 
                              nw_names=None, cmap='RdBu_r', vmin=None, vmax=None,cbar_title='Unknown', 
                            only_negatives=False, only_positives=False, rolling_win='No Win',rolling_mode='mean', 
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
    edge_selection = []
    for roi in roi_list:
        if scan_name is None:
            edge_selection = edge_selection + [e for e in list(data.edge.values) if roi in e ]
        else:
            edge_selection = edge_selection + [e for e in list(data[scan_name].edge.values) if roi in e ]
    
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
    if rolling_win != 'No Win':
        if rolling_mode == 'mean':
            this_scan_ets_df = this_scan_ets_df.rolling(rolling_win, center=True).mean()
        if rolling_mode == 'median':
            this_scan_ets_df = this_scan_ets_df.rolling(rolling_win, center=True).median()
        if rolling_mode == 'min':
            this_scan_ets_df = this_scan_ets_df.rolling(rolling_win, center=True).min()
        if rolling_mode == 'max':
            this_scan_ets_df = this_scan_ets_df.rolling(rolling_win, center=True).max()
    # Extract min, max values automatically if user provided none
    # ===========================================================
    if vmin is None:
        vmin = np.quantile(this_scan_ets_df.values.flatten(),0.025)
    if vmax is None:
        vmax = np.quantile(this_scan_ets_df.values.flatten(),0.975)
    
    # Get index per network & sort them
    # =================================
    nw_names    = list(np.unique([(e.split('-')[0]).split('_')[1] for e in data.edge.values]+[(e.split('-')[1]).split('_')[1] for e in data.edge.values]))
    num_nws     = len(nw_names)
    
    nw_in_roi_list = list(np.unique([r.split('_')[1] for r in roi_list]))
    sel_network = nw_in_roi_list[0]
    
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

    aux = aux.T
    aux.name = 'ETS'
    aux_melted = aux.reset_index().melt(id_vars='Time',var_name='Edges',value_name='ETS')
    scan_heatmap = aux_melted.hvplot.heatmap(x='Time',y='Edges',C='ETS',width=1800, height=500, cmap=cmap, xlabel='Time', ylabel='Showing %d edges' % Nedges, clim=(vmin,vmax), yticks=yticks_info,ylim=(0,Nedges), xlim=(-1,Nacqs-1)).opts(xrotation=90,colorbar_opts={'title':cbar_title}) #, colorbar_position='top')
    
    #scan_heatmap = aux.hvplot.heatmap(x='Time',y='Edges',C='ETS',width=1800, height=500, cmap=cmap, xlabel='Time', ylabel='Showing %d edges' % Nedges, clim=(vmin,vmax), yticks=yticks_info,ylim=(0,Nedges), xlim=(-1,Nacqs-1)).opts(xrotation=90,colorbar_opts={'title':cbar_title}) #, colorbar_position='top')
    
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
    
    # RSS Timeseries Across ROIs
    # ==========================
    if (show_time_rss==True):
        rssts            = np.sqrt(np.sum(this_scan_ets_df.reset_index(drop=True).T**2, axis=1))
        rssts.index.name ='Time'
        rssts.columns    = ['Time RSS']
        rssts.name       = 'Time RSS'
        if rssts_plot_ylim is None:
            rssts_plot       = rssts.hvplot(color='k', width=1800, height=200, ylabel='RSST').opts(yaxis='left')
        else:
            rssts_plot       = rssts.hvplot(color='k', width=1800, height=200, ylabel='RSST',ylim=rssts_plot_ylim).opts(yaxis='left')
        # Select ROI TS
        roi_ts_df = pd.DataFrame(roi_ts_xrs[scan_name[0:3]].sel(dt='E02',roi=roi_list).values, columns=roi_list)
        roi_ts_df.index.name='Time'
        roi_ts_df.columns.name='ROI TS'
        roi_ts_plot = roi_ts_df.hvplot(label=roi_list[0],c='b',width=1800, height=200,).opts(muted_alpha=0,yaxis='left')
        
        # End also plotting ROI
        
        if show_neural_event_annots:
            rssts_plot = rssts_plot * get_neural_event_hv_annots(schedule2evonsets_dict,this_scan_schedule,0.25)
        if show_hrf_event_annots:
            rssts_plot = rssts_plot * get_hrf_event_hv_annots(schedule2evonsets_dict,this_scan_schedule,this_scan_tap_label,hrf_thr=hrf_thr,alpha=0.25)
        rssts_plot.opts(legend_position='bottom')
        
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
        rssedge_plot = rssedge.hvplot.scatter(x=edge_rss_period, y='Edges', color='k', width=200, height=500, ylim=(0,Nedges-1), xlim=(min_x,max_x), hover_cols=['Edge_Name'],s=5, legend=False, cmap=nw_color_map, ylabel='RSS for %s Edges' % str(Nedges)).opts(line_width=1, xrotation=90)
    # Final layout formation
    # ======================
    if (show_time_rss==True) and (show_edge_rss==False):
        out = pn.Column(scan_heatmap_annot,rssts_plot, roi_ts_plot)
    elif (show_edge_rss==True) and (show_time_rss==False):
        out = pn.Row(scan_heatmap_annot,rssedge_plot)
    elif (show_edge_rss==True) and (show_time_rss==True):
        out = pn.Row(pn.Column(scan_heatmap_annot,rssts_plot, roi_ts_plot),rssedge_plot )# pn.Column(pn.Row(scan_heatmap_annot_fig,rssroi_fig), pn.Row(rssts_fig,None))
    else:
        out =  scan_heatmap_annot
    return out


pn.Row(control_column,get_edge_carpetplot_hrf_scan).show()

# + [markdown] editable=true slideshow={"slide_type": ""}
# ***
# ***
# ***
# -

sbj_run_select   = pn.widgets.NestedSelect(options=DATASET, layout=pn.Row, levels=[{'name':'Subject','width':100},{'name':'Run','width':100}])
denoising_select = pn.widgets.Select(name='Denoising',options=['MEICA','Basic'], width=100)
zscoring_select  = pn.widgets.Select(name='ETS Z-scrore',options=['Yes','No'], width=100)
tseg_select      = pn.widgets.Select(name='Scan Segment',options=['Whole'] + TASKS, width=200)
schedule_select  = pn.widgets.Select(name='Schedule', options=['Schedule01','Schedule02','Schedule03'], width=200)
nw_select        = pn.widgets.Select(name='Network', options=list(roi_info['Network'].unique()), width=200)
scan_card = pn.Card(sbj_run_select,pn.Row(denoising_select,zscoring_select),title='Data Selection', width=250)
@pn.depends(sbj_run_select,denoising_select,zscoring_select,tseg_select, nw_select)
def get_edge_carpetplot_hrf_scan(sbj_run, denoising,zscoring,tseg,nw):
    sbj = sbj_run['Subject']
    run = sbj_run['Run']
    return carpet_hvplot_ets_oneNW(ets_xrs, roi_info,(sbj,run,denoising,zscoring), 'E02', nw, vmin=-5, vmax=5, show_hrf_event_annots=True, show_time_rss=True, show_edge_rss=True, edge_rss_period=tseg, only_positives=True, cmap='RdBu_r', cbar_title='E02 | Scan Level')
@pn.depends(schedule_select,denoising_select,zscoring_select,tseg_select, nw_select)
def get_edge_carpetplot_hrf_schedule(schedule,denoising,zscoring, tseg,nw):
    md_input = get_avg_per_schedule(ets_xrs,schedule,denoising,zscoring)
    return carpet_hvplot_ets_oneNW(md_input, roi_info, None, 'E02', nw, show_hrf_event_annots=True, vmin=-5, vmax=5,show_time_rss=True, show_edge_rss=True, edge_rss_period=tseg, only_positives=True, cmap='RdBu_r', cbar_title='E02 | Group Level', rssts_plot_ylim=(0,250))


pn.Column(pn.pane.Markdown('# MEICA Denosing Data (Edge View)'),pn.Row(pn.Column(sbj_run_select,denoising_select, zscoring_select,pn.layout.Divider(),tseg_select,pn.layout.Divider(), schedule_select, nw_select),
       pn.Tabs(('Individual Scan',get_edge_carpetplot_hrf_scan),('Schedule Avrg.',get_edge_carpetplot_hrf_schedule)))).show()

out

# ***

# +
data = ets_xrs
scan_name=('P3SBJ07','Events01','MEICA','Yes')
data_type='E02'
roi_list=roi_select_dict['MUSI'][0:3]
nw_names=None
show_title=True
cmap='RdBu_r'
vmin=None
vmax=None
cbar_title='Unknown'
only_negatives=False
only_positives=False
rolling_win=None
show_time_rss=False
show_edge_rss=False
show_neural_event_annots=False
show_hrf_event_annots=False
hrf_thr=0.2
edge_rss_period='Whole'
rssts_plot_ylim=None
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
edge_selection = []
for roi in roi_list:
    if scan_name is None:
        edge_selection = edge_selection + [e for e in list(data.edge.values) if roi in e ]
    else:
        edge_selection = edge_selection + [e for e in list(data[scan_name].edge.values) if roi in e ]

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
nw_names    = list(np.unique([(e.split('-')[0]).split('_')[1] for e in data.edge.values]+[(e.split('-')[1]).split('_')[1] for e in data.edge.values]))
num_nws     = len(nw_names)

nw_in_roi_list = list(np.unique([r.split('_')[1] for r in roi_list]))
sel_network = nw_in_roi_list[0]

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
#scan_heatmap_annot_fig = hv.render(scan_heatmap_annot)
#scan_heatmap_annot_fig.legend.click_policy="hide"

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
    #rssts_fig        = hv.render(rssts_plot)
    #rssts_fig.legend.click_policy="hide"

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
    #rssedge_fig        = hv.render(rssedge_plot)
# Final layout formation
# ======================
if (show_time_rss==True) and (show_edge_rss==False):
    out = pn.Column(scan_heatmap_annot,rssts_plot)
elif (show_edge_rss==True) and (show_time_rss==False):
    out = pn.Row(scan_heatmap_annot,rssedge_plot)
elif (show_edge_rss==True) and (show_time_rss==True):
    out = pn.Row(pn.Column(scan_heatmap_annot,rssts_plot),rssedge_plot )# pn.Column(pn.Row(scan_heatmap_annot_fig,rssroi_fig), pn.Row(rssts_fig,None))
else:
    out =  scan_heatmap_annot
# -

# ## 4.3. ETS based on ME-PFM outputs (with & without Z-scoring)

mepfm_ets_xrs = xr.Dataset()
for sbj in DATASET.keys():
    for run in DATASET[sbj]:            
        for criteria in tqdm(['aic','bic'], desc='%s,%s' %(sbj,run)):
            for denoising in ['Basic','MEICA']:
                for model in ['R2ONLY','RHO_ZERO']:
                    file_path_noZscore = osp.join(PRCS_DATA_DIR,sbj,'D05_ets',f'{sbj}_{run}_ets.{denoising}_{criteria}_{model}.{ATLAS_NAME}.noZscore.nc')
                    file_path_yesZscore = osp.join(PRCS_DATA_DIR,sbj,'D05_ets',f'{sbj}_{run}_ets.{denoising}_{criteria}_{model}.{ATLAS_NAME}.yesZscore.nc')
                    mepfm_ets_xrs[(sbj,run,denoising,criteria,model,'No')] = xr.open_dataarray(file_path_noZscore).sel(tr=np.arange(0,210))
                    mepfm_ets_xrs[(sbj,run,denoising,criteria,model,'No')].attrs['schedule']  = scan2schedule_dict[sbj,run]
                    mepfm_ets_xrs[(sbj,run,denoising,criteria,model,'No')].attrs['tap_label'] = scan2hand_dict[sbj,run]
                    mepfm_ets_xrs[(sbj,run,denoising,criteria,model,'No')].attrs['denoising'] = denoising
                    mepfm_ets_xrs[(sbj,run,denoising,criteria,model,'No')].attrs['model']     = model
                    mepfm_ets_xrs[(sbj,run,denoising,criteria,model,'No')].attrs['criteria']  = criteria
                    mepfm_ets_xrs[(sbj,run,denoising,criteria,model,'No')].attrs['zscoring']  = 'No'
                    mepfm_ets_xrs[(sbj,run,denoising,criteria,model,'Yes')] = xr.open_dataarray(file_path_yesZscore).sel(tr=np.arange(0,210))
                    mepfm_ets_xrs[(sbj,run,denoising,criteria,model,'Yes')].attrs['schedule'] = scan2schedule_dict[sbj,run]
                    mepfm_ets_xrs[(sbj,run,denoising,criteria,model,'Yes')].attrs['tap_label'] = scan2hand_dict[sbj,run]
                    mepfm_ets_xrs[(sbj,run,denoising,criteria,model,'Yes')].attrs['denoising'] = denoising
                    mepfm_ets_xrs[(sbj,run,denoising,criteria,model,'Yes')].attrs['model']     = model
                    mepfm_ets_xrs[(sbj,run,denoising,criteria,model,'Yes')].attrs['criteria']  = criteria
                    mepfm_ets_xrs[(sbj,run,denoising,criteria,model,'Yes')].attrs['zscoring']  = 'Yes'

sbj_run_select   = pn.widgets.NestedSelect(options=DATASET, layout=pn.Row, levels=[{'name':'Subject','width':100},{'name':'Run','width':100}])
denoising_select = pn.widgets.Select(name='SPFM Input',options=['MEICA','Basic'], width=200)
tseg_select      = pn.widgets.Select(name='Scan Segment',options=['Whole'] + TASKS, width=200)
schedule_select  = pn.widgets.Select(name='Schedule', options=['Schedule01','Schedule02','Schedule03'], width=200)
nw_select        = pn.widgets.Select(name='Network', options=list(roi_info['Network'].unique()), width=200)
model_select     = pn.widgets.Select(name='PFM Model',options=['R2ONLY','RHO_ZERO'], width=200)
criteria_select  = pn.widgets.Select(name='PFM Criteria',options=['bic','aic'], width=200)
zscoring_select  = pn.widgets.Select(name='Zscoring for ETS',options=['Yes','No'],width=200)
@pn.depends(sbj_run_select,denoising_select,model_select,criteria_select,zscoring_select,tseg_select, nw_select)
def get_edge_carpetplot_hemo_scan(sbj_run, denoising, model, criteria, zscoring, tseg, nw):
    sbj = sbj_run['Subject']
    run = sbj_run['Run']
    return carpet_hvplot_ets_oneNW(mepfm_ets_xrs, roi_info,(sbj,run,denoising,criteria,model,zscoring), 'E02-DR2fit', nw, vmin=-5, vmax=5, show_hrf_event_annots=True, show_time_rss=True, show_edge_rss=True, edge_rss_period=tseg, only_positives=True, cbar_title='E02-DR2fit | Scan Level')
@pn.depends(schedule_select,denoising_select,model_select,criteria_select,zscoring_select,tseg_select, nw_select)
def get_edge_carpetplot_hemo_schedule(schedule,denoising,model,criteria,zscoring,tseg,nw):
    md_input = get_avg_per_schedule(mepfm_ets_xrs,schedule,denoising,sel_model=model,sel_criteria=criteria, sel_zscoring=zscoring)
    return carpet_hvplot_ets_oneNW(md_input, roi_info, None, 'E02-DR2fit', nw, vmin=-5, vmax=5, show_hrf_event_annots=True, show_time_rss=True, show_edge_rss=True, edge_rss_period=tseg, only_positives=True, cbar_title='E02-DR2fit | Group Level', rssts_plot_ylim=(0,250))


pn.Column(pn.pane.Markdown('# MEICA Denosing + Deconvolved Data (Edge View)'),pn.Row(pn.Column(sbj_run_select,denoising_select,model_select,criteria_select,zscoring_select,pn.layout.Divider(),tseg_select,pn.layout.Divider(), schedule_select, nw_select),
       pn.Tabs(('Individual Scan',get_edge_carpetplot_hemo_scan),('Schedule Avrg.',get_edge_carpetplot_hemo_schedule)))).show()

















# ## 4. Load All Outputs from ME-PFM

mepfm_ts_xrs = xr.Dataset()
for sbj in tqdm(DATASET.keys(),desc='Subject'):
    for run in DATASET[sbj]:
        for denoising in ['Basic','MEICA']:
            for criteria in ['bic','aic']:
                for model in ['R2ONLY','RHO_ZERO']:
                    data_path = osp.join(PRCS_DATA_DIR,sbj,'D06_MEPFM',f'{sbj}_{run}_MEPF.{denoising}_{criteria}_{model}.{ATLAS_NAME}.nc')
                    if not osp.exists(data_path):
                        mepfm_ts_xrs[(sbj,run,denoising,criteria,model)] = None
                        print("++ WARNING: Missing file [%s]" % data_path)
                        continue
                    else:
                        mepfm_ts_xrs[(sbj,run,denoising,criteria,model)] = xr.open_dataarray(data_path).sel(tr=np.arange(0,210))
                        mepfm_ts_xrs[(sbj,run,denoising,criteria,model)].attrs['schedule']  = scan2schedule_dict[sbj,run]
                        mepfm_ts_xrs[(sbj,run,denoising,criteria,model)].attrs['tap_label'] = scan2hand_dict[sbj,run]
                        mepfm_ts_xrs[(sbj,run,denoising,criteria,model)].attrs['denoising'] = denoising
                        mepfm_ts_xrs[(sbj,run,denoising,criteria,model)].attrs['criteria'] = criteria
                        mepfm_ts_xrs[(sbj,run,denoising,criteria,model)].attrs['model'] = model

# ## 4.2. Compute DR2 ETS (but do only_negatives and rolling win first)

from itertools import combinations
from utils.ets import get_ets

# +
# %%time
dr2_ets        = xr.Dataset()
only_negatives = True
rolling_win    = 2

for sbj in DATASET.keys():
    for run in tqdm(DATASET[sbj],desc=sbj):
        for denoising in ['Basic','MEICA']:
            for criteria in ['bic','aic']:
                for model in ['R2ONLY','RHO_ZERO']: 
                    # Load ROI Timeseries
                    in_ts = mepfm_ts_xrs[(sbj,run,denoising,criteria,model)]
            
                    #dt_coords = list(in_ts.coords['dt'].values)
                    tr_coords = list(in_ts.coords['tr'].values)
                    roi_names = list(in_ts.coords['roi'].values)
                    # Create Empty ETS Xarray
                    edge_list_tuples = [e for e in combinations(roi_names,2)]
                    edge_coords      = ['-'.join(e) for e in edge_list_tuples]        
                    out_xr = xr.DataArray(dims=['dt','tr','edge'],
                                          coords = {'dt':['DR2'],
                                                    'tr':tr_coords,
                                                    'edge':edge_coords},name='ets')
                    
                    # Remove positives
                    if only_negatives:
                        in_ts = xr.where(in_ts > 0, 0, in_ts)

                    # Focus on DR2
                    in_ts = in_ts.sel(dt='DR2')
                    
                    # Apply rolling window
                    if rolling_win is not None:
                        in_ts = in_ts.rolling(tr=rolling_win, center=False).min()
                        in_ts = in_ts.fillna(0)

                    # Compute ETS
                    out_xr.loc['DR2'] = get_ets(in_ts, normalize=False)
            
                    # Write ETS Xarray to disk
                    dr2_ets[(sbj,run,denoising,criteria,model)]                    = out_xr
                    dr2_ets[(sbj,run,denoising,criteria,model)].attrs['schedule']  = scan2schedule_dict[sbj,run]
                    dr2_ets[(sbj,run,denoising,criteria,model)].attrs['tap_label'] = scan2hand_dict[sbj,run]
                    dr2_ets[(sbj,run,denoising,criteria,model)].attrs['denoising'] = denoising
                    dr2_ets[(sbj,run,denoising,criteria,model)].attrs['criteria']  = criteria
                    dr2_ets[(sbj,run,denoising,criteria,model)].attrs['model']     = model

# +
sbj_run_select   = pn.widgets.NestedSelect(options=DATASET, layout=pn.Row, levels=[{'name':'Subject','width':100},{'name':'Run','width':100}])
denoising_select = pn.widgets.Select(name='SPFM Input',options=['MEICA','Basic'], width=200)
model_select     = pn.widgets.Select(name='PFM Model',options=['R2ONLY','RHO_ZERO'], width=200)
criteria_select  = pn.widgets.Select(name='PFM Criteria',options=['bic','aic'], width=200)
tseg_select      = pn.widgets.Select(name='Scan Segment',options=['Whole'] + TASKS, width=200)
schedule_select = pn.widgets.Select(name='Schedule', options=['Schedule01','Schedule02','Schedule03'], width=200)
nw_select        = pn.widgets.Select(name='Network', options=list(roi_info['Network'].unique()), width=200)

@pn.depends(sbj_run_select,denoising_select,model_select, criteria_select, tseg_select, nw_select)
def get_edge_carpetplot_neuro_scan(sbj_run, denoising, model, criteria, tseg,nw):
    sbj = sbj_run['Subject']
    run = sbj_run['Run']
    return carpet_hvplot_ets_oneNW(dr2_ets, roi_info,(sbj,run,denoising,criteria,model), 'DR2', nw, show_neural_event_annots=True, show_time_rss=True, show_edge_rss=True, edge_rss_period=tseg, only_positives=True, cmap='gray_r')
@pn.depends(schedule_select,denoising_select,model_select, criteria_select,tseg_select, nw_select)
def get_edge_carpetplot_neuro_schedule(schedule,denoising,model, criteria, tseg,nw):
    md_input = get_avg_per_schedule(dr2_ets,schedule,denoising,sel_model=model,sel_criteria=criteria)
    return carpet_hvplot_ets_oneNW(md_input, roi_info, None, 'DR2', nw, show_neural_event_annots=True, show_time_rss=True, show_edge_rss=True, edge_rss_period=tseg, only_positives=True, cmap='gray_r')


# -

pn.Row(pn.Column(sbj_run_select,denoising_select,model_select, criteria_select, pn.layout.Divider(),tseg_select,pn.layout.Divider(), schedule_select, nw_select),
       pn.Tabs(('Individual Scan',get_edge_carpetplot_neuro_scan),('Schedule Avrg.',get_edge_carpetplot_neuro_schedule))).show()

dr2_ets[('P3SBJ07','Events01','MEICA','bic','RHO_ZERO')].max()











def plot_edge_carpet(data, scan_name, data_type, 
                     show_title=True, cmap='RdBu_r', vmin=None, vmax=None,
                     sorted_edges=None,cbar_title='Unknown', only_positives=False, only_negatives=False, show_rss_ts=False, show_rss_edge=False, sig_spikes=None, show_neural_event_annots=False, show_hrf_event_annots=False,hrf_thr=0.2):
    
    # BASIC DATA EXTRACTION AND CHARACTERIZATION
    # ===========================================
    # If a specific xr.Dataarray is specified, go ahead and select it
    if scan_name is not None:
        data = data[scan_name]
    # If a specific datatype (i.e., value for the dt dimension) is specified, go ahead and select it.
    if data_type is not None:
        data = data.sel(dt=data_type)
    # If a list of sorted edges is provided, go ahead and apply it (assumption here is that the second remaining dimension is edge)
    if sorted_edges is not None:
        data = data.loc[:,sorted_edges]
    # Extract information about the schedule, number of acquisitions and number of edges
    SCH                = data.schedule 
    TAP_LABEL          = data.tap_label
    Nacqs,Nedges       = data.shape

    # DATA TRANSFORMATIONS PRIOR TO PLOTTING
    # ======================================
    # Remove positive values, if selected
    if only_negatives:
        data = xr.where(data < 0, data, 0)
    # Remove negative values, if selected
    if only_positives:
        data = xr.where(data > 0, data, 0)
    # Maybe cap the data?

    # PLOT CONFIGURATION
    # ==================
    # Computing Max and Min
    if vmin is None:
        vmin = data.quantile(.025).values
    if vmax is None:
        vmax = data.quantile(.975).values
    #clim = np.array([abs(vmin), abs(vmax)]).max()

    # Remove strings from edge dimension
    edge_mapping        = {k: v for v, k in enumerate(data.coords['edge'].values)}
    data.coords['edge'] = [edge_mapping[x] for x in data.coords['edge'].values]

    # Create Figure structure
    fig = plt.figure(figsize=(20,10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[8, 2], width_ratios=[9,1])

    # Create carpet plot
    # ==================
    ax1          = fig.add_subplot(gs[0])
    sns_heatmap  = sns.heatmap(data.T, cmap=cmap, ax=ax1, vmin=vmin, vmax=vmax, cbar=False)
    
    # Annotate Carpet with Task Info
    # ==============================
    if show_neural_event_annots:
        task_onsets, task_offsets = {},{}
        for task in TASKS:      
            task_onsets[task]  = schedule2evonsets_dict[(SCH,task)]/2
            task_offsets[task] = task_onsets[task]+2
    if show_hrf_event_annots:
        task_onsets, task_offsets = get_hrf_event_onsets_offsets(SCH,TAP_LABEL,hrf_thr)

    if show_neural_event_annots or show_hrf_event_annots:
        for task in TASKS:
            for onset,offset in zip(task_onsets[task],task_offsets[task]):
                ax1.axvspan(onset,offset,color=TASK_COLORS[task],alpha=0.3)
    ax1.set_ylabel='Connections'
    ax1.set_xlabel='Time'

     
    # Remove x-axis labels and ticks from the first plot to avoid clutter
    plt.setp(ax1.get_xticklabels(), visible=False)

    # Adjust the subplots to align the x-axes without the colorbar affecting the layout
    gs.update(left=0.1, right=0.85, top=0.9, bottom=0.1, hspace=0.05)
    plt.close()
    
    if show_rss_ts:
        # Create carpet plot
        # ==================
        ax2          = fig.add_subplot(gs[2])
        rss_ts       = np.sqrt(np.sum(data**2, axis=1))
        pd.DataFrame(rss_ts.values.T).plot(ax=ax2, xlim=(0,Nacqs),c='k', linewidth=0.5)
    if show_rss_edge:
        # Create carpet plot
        # ==================
        ax3          = fig.add_subplot(gs[1])
        rss_edge      = np.sqrt(np.sum(data.T**2, axis=1)).T
        rss_edge      = pd.DataFrame(rss_edge.values,columns=['rss_edge'])
        rss_edge.index.name = 'edge'
        x            = rss_edge['rss_edge'].values
        y            = rss_edge.reset_index()['edge'].values
        ax3.plot(x,-y,c='k', linewidth=0.5)
        ax3.set_ylim(-Nedges,0)
        ax3.set_yticks([])
        ax3.set_yticklabels([])
    # Color bar
    if show_rss_edge:
        # Create colorbar for the carpet plot.
        cbar_ax = fig.add_axes([ax3.get_position().x1+0.02, ax3.get_position().y0, 0.02, ax3.get_position().height])
    else:
        # Create colorbar for the carpet plot.
        cbar_ax = fig.add_axes([ax1.get_position().x1+0.02, ax1.get_position().y0, 0.02, ax1.get_position().height])
    fig.colorbar(ax1.collections[0], cax=cbar_ax)
    if sig_spikes is not None:
        for y in sig_spikes:
            ax1.axvline(y, color='k')
    
    return fig

plot_edge_carpet(ets_xrs, (sbj,run,denoising,zscoring), 'E02', 
                     show_title=True, cmap='RdBu_r', vmin=None, vmax=None,
                     cbar_title='Unknown', only_positives=False, only_negatives=False, show_rss_ts=False, show_rss_edge=True, sig_spikes=None, show_hrf_event_annots=True,sorted_edges=[e for e in list(ets_xrs[(sbj,run,'Basic','No')].edge.values) if (('DorsAttnA' in e) and ('VisCent' in e)) ] )

pk_inds, pk_amp, numpk, pval, pcnt = {}, {}, {}, {}, {}
for sbj in tqdm(DATASET.keys(),desc='Subject'):
    for run in DATASET[sbj]:
        for denoising in ['Basic','MEICA']:
            pk_inds[(sbj,run,denoising)], pk_amp[(sbj,run,denoising)], numpk[(sbj,run,denoising)], pval[(sbj,run,denoising)], pcnt[(sbj,run,denoising)] = detect_RSSevents(roi_ts_xrs[(sbj,run,denoising)].sel(dt='E02').values)



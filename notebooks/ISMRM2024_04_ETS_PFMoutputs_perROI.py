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

# + [markdown] panel-layout={"width": 100, "height": 93.5653, "visible": true}
# # Description
#
# This notebook shows ETS carpet plots for inputs to PFM both at the single-scan and group level

# + editable=true slideshow={"slide_type": ""} panel-layout={"width": 100, "height": 0, "visible": true}
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

# + [markdown] panel-layout={"width": 100, "height": 60.5966, "visible": true}
# # 1. Load ROI Info
# -

roi_info, roi_names = get_roi_info(ATLAS_DIR, ATLAS_NAME)

# + [markdown] panel-layout={"width": 100, "height": 60.5966, "visible": true}
# # 2. Load Paradigm Info
# -

scan2schedule_dict, scan2hand_dict, schedule2evonsets_dict, _ = get_paradigm_info(DATASET)

# + [markdown] panel-layout={"width": 100, "height": 60.5966, "visible": true}
# # 3. Load Per-task Top ROIs (according to NeuroSynth)
# -

atlas_path = osp.join(ATLAS_DIR,f'{ATLAS_NAME}.nii.gz')
nlm = NiftiLabelsMasker(atlas_path )

for task in TASKS:
    group_spm_path = osp.join(ATLAS_DIR,f'ALL.ttest.mACF.{task}.DN.resampled.nii.gz')
    aux = nlm.fit_transform(group_spm_path)
    roi_info[task] = aux.flatten()

# + editable=true slideshow={"slide_type": ""} panel-layout={"width": 100, "height": 170.114, "visible": true}
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

# + [markdown] panel-layout={"width": 100, "height": 93.5653, "visible": true}
# # 4. Load ROI Timeseries
#
# > NOTE: So we can compare ETS to originating ROI Timeseries
# -

roi_ts_xrs = xr.Dataset()
for sbj in tqdm(DATASET.keys(),desc='Subject'):
    for run in DATASET[sbj]:
        for denoising in ['Basic','MEICA']:
            for criteria in ['bic','aic']:
                for model in ['R2ONLY','RHO_ZERO']:
                    data_path = osp.join(PRCS_DATA_DIR,sbj,'D06_MEPFM',f'{sbj}_{run}_MEPF.{denoising}_{criteria}_{model}.{ATLAS_NAME}.nc')
                    if not osp.exists(data_path):
                        roi_ts_xrs[(sbj,run,denoising,criteria,model)] = None
                        print("++ WARNING: Missing file [%s]" % data_path)
                        continue
                    else:
                        roi_ts_xrs[(sbj,run,denoising,criteria,model)] = xr.open_dataarray(data_path)#.sel(tr=np.arange(0,210))
                        roi_ts_xrs[(sbj,run,denoising,criteria,model)].attrs['schedule']  = scan2schedule_dict[sbj,run]
                        roi_ts_xrs[(sbj,run,denoising,criteria,model)].attrs['tap_label'] = scan2hand_dict[sbj,run]
                        roi_ts_xrs[(sbj,run,denoising,criteria,model)].attrs['denoising'] = denoising
                        roi_ts_xrs[(sbj,run,denoising,criteria,model)].attrs['criteria']  = criteria
                        roi_ts_xrs[(sbj,run,denoising,criteria,model)].attrs['model']     = model

# + [markdown] panel-layout={"width": 100, "height": 101.676, "visible": true} editable=true slideshow={"slide_type": ""}
# ***
# # 5. Load ETS-level Data
#
# ## 5.1. ETS on PFM outputs
# -

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

available_traces = list(mepfm_ets_xrs.coords['dt'].values)

available_traces_types = {t:'BOLD' if 'fit' in t else 'Neural' for t in available_traces}
cplot_colormaps        = {'BOLD':'RdBu_r','Neural':'PiYG_r'}
cplot_minmax           = {'BOLD':(-2,2),'Neural':(-2,2)}

# + [markdown] editable=true slideshow={"slide_type": ""} panel-layout={"width": 100, "height": 60.5966, "visible": true}
# # 6. Plot ETS at the scan level for PFM inputs

# +
# Data Selection
sbj_run_select   = pn.widgets.NestedSelect(options=DATASET, layout=pn.Row, levels=[{'name':'Subject','width':100},{'name':'Run','width':100}])
trace_select      = pn.widgets.Select(name='Trace',options=available_traces,value='E02-DR2fit', width=150,description='3dPFM trace to plot')
denoising_select = pn.widgets.Select(name='Denoising',options=['MEICA','Basic'], width=100)
zscoring_select  = pn.widgets.Select(name='ETS Z-scrore',options=['Yes','No'], width=100, description='Whether or not to Z-score ROI timeseries prior to ETS compuation')
criteria_select  = pn.widgets.Select(name='Criteria',options=['bic','aic'], width=100, description='Criteria used in 3dMEPFM')
model_select     = pn.widgets.Select(name='Model',options=['R2ONLY','RHO_ZERO'], width=100, description='Model used in 3dMEPFM')

scan_card        = pn.Card(sbj_run_select,pn.Row(denoising_select,zscoring_select),pn.Row(criteria_select,model_select),trace_select, title='Data Selection', width=250)

# Carpet Plot Configuration
only_positives_check = pn.widgets.Checkbox(name='Show only positive values')
only_negatives_check = pn.widgets.Checkbox(name='Show only negative values')
cplot_config_card    = pn.Card(only_positives_check,only_negatives_check,title='CPlot Configuration', width=250)

tseg_select         = pn.widgets.Select(name='Scan Segment Selection',options=['All'] + TASKS, width=200, description='Select the scan segment you want to be used for computing the ROI-RS scatter on the right')
rssts_config_card   = pn.Card(tseg_select, title='ROI-TS Configuration', width=250)

control_column = pn.Column(pn.layout.Divider(), scan_card, pn.layout.Divider(), roi_sel_card, pn.layout.Divider(), cplot_config_card,pn.layout.Divider(),rssts_config_card,pn.layout.Divider() )


# + editable=true slideshow={"slide_type": ""}
@pn.depends(sbj_run_select, denoising_select, trace_select, zscoring_select, model_select,criteria_select,
            only_positives_check, only_negatives_check,
            tseg_select, roi_list_multiselect)
def options_to_plot(sbj_run, denoising, trace, zscoring, model, criteria, only_positives, only_negatives, time_segment, seed_roi):
    assert isinstance(seed_roi,str), "++ ERROR[options_to_plot]: seed_rois is not a string"
    sbj = sbj_run['Subject']
    run = sbj_run['Run']
    scan_name = (sbj,run,denoising,criteria,model,zscoring)
    # Edge Selection
    all_edges = [tuple(e.split('-')) for e in list(mepfm_ets_xrs[scan_name].edge.values)]
    edge_selection = ['-'.join(e) for e in all_edges if seed_roi==e[0] or seed_roi==e[1]]
    data = mepfm_ets_xrs[scan_name].sel(dt=trace,edge=edge_selection)
    
    # Remove positive or regative values
    aux_attrs = data.attrs
    if only_positives:
        data = xr.where(data > 0, data, np.nan)
    if only_negatives:
        data = xr.where(data < 0, data, np.nan)
    data.attrs = aux_attrs

    # Get seed ROI timeseries
    roi_ts = roi_ts_xrs[scan_name[0:5]].sel(dt=trace,roi=seed_roi).to_dataframe()
    roi_ts.drop(['roi','dt'],axis=1, inplace=True)
    roi_ts.columns=[available_traces_types[trace]]
    roi_ts.reset_index(drop=True,inplace=True)
    roi_ts['Time [sec]'] = 2.0 * np.arange(roi_ts.shape[0])
    out = create_ets_figure(data,available_traces_types[trace],2.0,'tr','edge',seed_roi,trace,roi_ts,
                      width=2000, height=500, cmap=cplot_colormaps[available_traces_types[trace]], 
                      vmin=cplot_minmax[available_traces_types[trace]][0], vmax=cplot_minmax[available_traces_types[trace]][1], 
                      rssts_min=None, rssts_max=None, 
                      roits_min=None, roits_max=None,
                      time_segment=time_segment, hrf_thr=0.2)
    return out


# + editable=true slideshow={"slide_type": ""}
dashboard1 = pn.Row(control_column,options_to_plot)

# + editable=true slideshow={"slide_type": ""} panel-layout={"width": 100, "height": 930, "visible": true}
dashboard1

# + [markdown] panel-layout={"width": 100, "height": 60.5966, "visible": true}
# # 7. Plot ETS at the Schedule Level (Inputs to PFM)
# -

cplot_minmax           = {'BOLD':(-2,2),'Neural':(-1,1)}

# + editable=true slideshow={"slide_type": ""}
schedule_select      = pn.widgets.Select(name='Schedule',options=['Schedule01','Schedule02','Schedule03'], width=125)
denoising_select     = pn.widgets.Select(name='Denoise',options=['MEICA','Basic'], width=100)
trace_select         = pn.widgets.Select(name='Trace',options=available_traces,value='E02-DR2fit', width=100)
zscoring_select      = pn.widgets.Select(name='ETS Z-scrore',options=['Yes','No'], width=100, description='Whether or not to Z-score ROI timeseries prior to ETS compuation')
criteria_select      = pn.widgets.Select(name='Criteria',options=['bic','aic'], width=75, description='Criteria used in 3dMEPFM')
model_select         = pn.widgets.Select(name='Model',options=['R2ONLY','RHO_ZERO'], width=100, description='Model used in 3dMEPFM')

rwin_mode_cplot_select   = pn.widgets.Select(name='Carpet Roll-Win Function',options=['mean','median','min','max','count'], width=200)
rwin_dur_cplot_select    = pn.widgets.IntInput(name='Carpet Roll-Win Duration',start=0,end=10,step=1,value=0, width=200)
sch_only_positives_check = pn.widgets.Checkbox(name='Remove neg. values prior to avg')
sch_only_negatives_check = pn.widgets.Checkbox(name='Remove pos. values prior to avg')
value_cap_input          = pn.widgets.FloatInput(name='Cap value', value=10., width=200, description='Cap value prior to averaging. Entires with absolute values above this value will be capped to the value (keeping the correct sign)')

tseg_select             = pn.widgets.Select(name='Scan Segment Selection',options=['All'] + TASKS, width=200, description='Select the scan segment you want to be used for computing the ROI-RS scatter on the right')

schedule_selection_card = pn.Card(pn.Row(schedule_select,criteria_select),pn.Row(zscoring_select,denoising_select),pn.Row(model_select,trace_select),title='Schedule Selection', width=250)
schedule_avg_card       = pn.Card(sch_only_positives_check, sch_only_negatives_check, pn.layout.Divider(), rwin_mode_cplot_select, rwin_dur_cplot_select, pn.layout.Divider(), value_cap_input, title='Schedule Avg. Configuration', width=250)
cplot_config_card       = pn.Card(only_positives_check,only_negatives_check,title='CPlot Configuration', width=250)
rssts_config_card       = pn.Card(tseg_select, title='ROI-TS Configuration', width=250)

sch_control_col = pn.Column(pn.layout.Divider(),schedule_selection_card,pn.layout.Divider(),roi_sel_card,pn.layout.Divider(), schedule_avg_card,pn.layout.Divider(),rssts_config_card,pn.layout.Divider())


# + editable=true slideshow={"slide_type": ""}
def _get_ets_schedule_avg(data, schedule, denoising, zscoring, model=None, criteria=None, 
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
    zscoring: selected zscoring method
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
    if (model is None) & (criteria is None):
        pfm_outputs = False
    else:
        pfm_outputs = True
        
    # Select dataarrays of interest from dataset provided as input
    if pfm_outputs:
        selected_dataarrays = data.filter_by_attrs(schedule=schedule, denoising=denoising, zscoring=zscoring, model=model, criteria=criteria)
    else:
        selected_dataarrays = data.filter_by_attrs(schedule=schedule, denoising=denoising, zscoring=zscoring)
        
    # Capping input data (if requested)
    if values_cap is not None:
        for da_name,da in selected_dataarrays.items():
            selected_dataarrays[da_name] = xr.where(da < -values_cap , -values_cap, da)
        for da_name,da in selected_dataarrays.items():
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


# + editable=true slideshow={"slide_type": ""}
@pn.depends(schedule_select, denoising_select, zscoring_select, model_select, criteria_select, trace_select, roi_list_multiselect,
            rwin_mode_cplot_select,rwin_dur_cplot_select, sch_only_positives_check, sch_only_negatives_check, value_cap_input,
            tseg_select)
def sch_options_to_plot(schedule, denoising, zscoring, model, criteria, trace, seed_roi,
                        rwin_mode,rwin_dur, only_positives, only_negatives, values_cap,
                        time_segment):
    # Create Seed ROI Timseries average
    roi_ts_this_schedule = roi_ts_xrs.filter_by_attrs(schedule=schedule, denoising=denoising, model=model, criteria=criteria).sel(dt=trace,roi=seed_roi)
    roi_ts_this_schedule = xr.concat([roi_ts_this_schedule[name] for name in roi_ts_this_schedule.data_vars], dim='scan')
    roi_ts_this_schedule = roi_ts_this_schedule.mean(dim='scan')
    roi_ts_this_schedule = roi_ts_this_schedule.to_dataframe()
    roi_ts_this_schedule.drop(['roi','dt'], axis=1, inplace=True)
    roi_ts_this_schedule.columns=[available_traces_types[trace]]
    roi_ts_this_schedule.reset_index(drop=True,inplace=True)
    roi_ts_this_schedule['Time [sec]'] = np.arange(roi_ts_this_schedule.shape[0])*2.0
    # Edge Selection
    all_edges = [tuple(e.split('-')) for e in list(mepfm_ets_xrs.edge.values)]
    edge_selection = ['-'.join(e) for e in all_edges if seed_roi==e[0] or seed_roi==e[1]]
    
    # Create Schedule carpet plot
    sch_avg_data = _get_ets_schedule_avg(mepfm_ets_xrs.sel(dt=trace,edge=edge_selection), schedule, denoising, zscoring, model=model, criteria=criteria, 
                            only_negatives=only_negatives, only_positives=only_positives, 
                            rwin_dur=rwin_dur, rwin_mode=rwin_mode,
                            values_cap = values_cap) 
    out = create_ets_figure(sch_avg_data,available_traces_types[trace],2.0,'tr','edge',seed_roi,trace,roi_ts_this_schedule,
                      width=2000, height=500, cmap=cplot_colormaps[available_traces_types[trace]], 
                      vmin=cplot_minmax[available_traces_types[trace]][0], vmax=cplot_minmax[available_traces_types[trace]][1],
                      rssts_min=None, rssts_max=None, 
                      roits_min=None, roits_max=None,
                      time_segment=time_segment, hrf_thr=0.2)
    
    return out


# + editable=true slideshow={"slide_type": ""}
dashboard2 = pn.Row(sch_control_col,sch_options_to_plot)

# + editable=true slideshow={"slide_type": ""} panel-layout={"width": 100, "height": 10, "visible": true}
dashboard2

# + editable=true slideshow={"slide_type": ""}

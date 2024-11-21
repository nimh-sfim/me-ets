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

# + [markdown] editable=true slideshow={"slide_type": ""} panel-layout={"width": 100, "height": 188.778, "visible": true}
# ***
# # Description
#
# This notebook loads ROI timeseries that result from running ME-PFM at the ROI level.
#
# It then creates two dashboards to explore them as carpet plots. 
#
# Dashboard one allows the exploration of data at the individual scan level
#
# Dashboard two allows the exploration of data at the group level, as a function of task schedule.

# + panel-layout={"width": 100, "height": 0, "visible": true} editable=true slideshow={"slide_type": ""}
import sys
sys.path.append('../python/')

# + editable=true slideshow={"slide_type": ""}
from utils.basics import PRCS_DATA_DIR, SCHEDULES_DIR, DATASET, get_paradigm_info, get_roi_info, TASKS
from utils.basics import ATLAS_DIR_17NETWORKS as ATLAS_DIR
from utils.basics import ATLAS_NAME_17NETWORKS as ATLAS_NAME

from utils.plotting import create_roits_figure, _get_roits_schedule_avg

# + panel-layout={"width": 100, "height": 0, "visible": true} editable=true slideshow={"slide_type": ""}
import os.path as osp
import xarray as xr
import numpy as np
import pandas as pd
from tqdm import tqdm
import panel as pn
pn.extension()
# -

from nilearn.maskers import NiftiLabelsMasker

# + panel-layout={"width": 100, "height": 0, "visible": true}
# allows visualisation in notebook
from bokeh.io import output_notebook
from bokeh.resources import INLINE
output_notebook(INLINE)

# + [markdown] panel-layout={"width": 100, "height": 60.5966, "visible": true}
# # 1. Load ROI Info
# -

roi_info, roi_names = get_roi_info(ATLAS_DIR, ATLAS_NAME)

scan2schedule_dict, scan2hand_dict, schedule2evonsets_dict, _ = get_paradigm_info(DATASET)

# + [markdown] panel-layout={"width": 100, "height": 60.5966, "visible": true} editable=true slideshow={"slide_type": ""}
# # 2. Load Paradigm Info

# + [markdown] panel-layout={"width": 100, "height": 60.5966, "visible": true}
# # 3. Load Per-task Top ROIs (according to NeuroSynth)
# -

atlas_path = osp.join(ATLAS_DIR,f'{ATLAS_NAME}.nii.gz')
nlm = NiftiLabelsMasker(atlas_path )

for task in TASKS:
    group_spm_path = osp.join(ATLAS_DIR,f'ALL.ttest.mACF.{task}.DN.resampled.nii.gz')
    aux = nlm.fit_transform(group_spm_path)
    roi_info[task] = aux.flatten()

# + [markdown] panel-layout={"width": 100, "height": 135.241, "visible": true} editable=true slideshow={"slide_type": ""}
# ***
# # 4. Load ROI Level Data
#
# ## 4.1. Load PFM outputs
#
# Load ROI Timeseries for E01, E02, E03 & E01-MEICA, E02-MEICA and E03-MEICA
# -

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
                        mepfm_ts_xrs[(sbj,run,denoising,criteria,model)] = xr.open_dataarray(data_path)#.sel(tr=np.arange(0,210))
                        mepfm_ts_xrs[(sbj,run,denoising,criteria,model)].attrs['schedule']  = scan2schedule_dict[sbj,run]
                        mepfm_ts_xrs[(sbj,run,denoising,criteria,model)].attrs['tap_label'] = scan2hand_dict[sbj,run]
                        mepfm_ts_xrs[(sbj,run,denoising,criteria,model)].attrs['denoising'] = denoising
                        mepfm_ts_xrs[(sbj,run,denoising,criteria,model)].attrs['criteria']  = criteria
                        mepfm_ts_xrs[(sbj,run,denoising,criteria,model)].attrs['model']     = model

# + panel-layout={"width": 100, "height": 244.787, "visible": true}
mepfm_ts_xrs
# -

available_traces = list(mepfm_ts_xrs.coords['dt'].values)

available_traces_types = {t:'BOLD' if 'fit' in t else 'Neural' for t in available_traces}
cplot_colormaps        = {'BOLD':'RdBu_r','Neural':'gray'}

# + [markdown] panel-layout={"width": 100, "height": 68.4091, "visible": true}
# ***
# # 4. Plot scan-level ROI Timeseries (Basic and MEICA denoised | Separate echoes)

# + editable=true slideshow={"slide_type": ""}
sbj_run_select       = pn.widgets.NestedSelect(options=DATASET, layout=pn.Row, levels=[{'name':'Subject','width':100},{'name':'Run','width':100}])
denoising_select     = pn.widgets.Select(name='SPFM Input',options=['MEICA','Basic'], width=100)
model_select         = pn.widgets.Select(name='PFM Model',options=['R2ONLY','RHO_ZERO'], width=100)
criteria_select      = pn.widgets.Select(name='PFM Criteria',options=['bic','aic'], width=100)
trace_select         = pn.widgets.Select(name='Trace',options=available_traces,value='E02-DR2fit', width=100)
top_roi_task_select  = pn.widgets.Select(name='ROI Selection (top 50)',options=['All ROIs']+TASKS, width=200, description='ROI Selection for Carpet plot. All ROIs or TOP 50 active ROIs for a given task')

only_positives_check = pn.widgets.Checkbox(name='Show only positive values')
only_negatives_check = pn.widgets.Checkbox(name='Show only negative values')
cplot_min            = pn.widgets.FloatSlider(name='CBar Min. Value', start=-5, end=0, step=0.01, value=-0.05, width=200)
cplot_max            = pn.widgets.FloatSlider(name='CBar Max. Value', start=0, end=5, step=0.01, value=0.05, width=200)
tseg_select          = pn.widgets.Select(name='Scan Segment Selection',options=['All'] + TASKS, width=200, description='Select the scan segment you want to be used for computing the ROI-RS scatter on the right')

scan_selection_card = pn.Card(sbj_run_select,pn.Row(denoising_select, model_select), pn.Row(criteria_select, trace_select),title='Scan Selection', width=250)
cplot_config_card   = pn.Card(top_roi_task_select, pn.layout.Divider(), only_positives_check,only_negatives_check,pn.layout.Divider(), cplot_min, cplot_max, title='CPlot Configuration', width=250)
rssts_config_card   = pn.Card(tseg_select, title='ROI-TS Configuration', width=250)

control_col = pn.Column(pn.layout.Divider(),scan_selection_card,pn.layout.Divider(),cplot_config_card,pn.layout.Divider(),rssts_config_card,pn.layout.Divider())


# + editable=true slideshow={"slide_type": ""}
@pn.depends(sbj_run_select, denoising_select, model_select, criteria_select, trace_select, top_roi_task_select, 
            only_positives_check, only_negatives_check, cplot_min, cplot_max,
            tseg_select)
def options_to_plot(sbj_run, denoising, model, criteria, trace, task, only_positives, only_negatives, vmin, vmax, time_segment):
    sbj = sbj_run['Subject']
    run = sbj_run['Run']
    if task == 'All ROIs':
        data = mepfm_ts_xrs[sbj,run,denoising, criteria, model].sel(dt=trace)
    else:
        rois = list(roi_info.sort_values(task,ascending=False)['ROI_Name'])[0:50]
        data = mepfm_ts_xrs[sbj,run,denoising, criteria, model].sel(dt=trace, roi=rois)
    
    aux_attrs = data.attrs
    if only_positives:
        data = xr.where(data > 0, data, np.nan)
    if only_negatives:
        data = xr.where(data < 0, data, np.nan)
    data.attrs = aux_attrs
    
    out = create_roits_figure(data,available_traces_types[trace],2.0,'tr','roi',
                      width=2000, height=500, cmap=cplot_colormaps[available_traces_types[trace]], 
                      vmin=vmin, vmax=vmax, 
                      rssts_min=None, rssts_max=None, 
                      roits_min=None, roits_max=None,
                      time_segment=time_segment, hrf_thr=0.2)
    return out


# + panel-layout={"width": 100, "height": 10, "visible": true}
dashboard1 = pn.Row(control_col,options_to_plot)

# + editable=true slideshow={"slide_type": ""}
dashboard1

# + [markdown] panel-layout={"width": 100, "height": 68.4091, "visible": true}
# ***
# # 5. Plot Schedule-level ROI Timeseries (Basic and MEICA denoised | Separate echoes)

# + editable=true slideshow={"slide_type": ""}
schedule_select      = pn.widgets.Select(name='Schedule',options=['Schedule01','Schedule02','Schedule03'], width=150)
denoising_select     = pn.widgets.Select(name='SPFM Input',options=['MEICA','Basic'], width=100)
model_select         = pn.widgets.Select(name='PFM Model',options=['R2ONLY','RHO_ZERO'], width=100)
criteria_select      = pn.widgets.Select(name='PFM Criteria',options=['bic','aic'], width=100)
trace_select         = pn.widgets.Select(name='Trace',options=available_traces,value='E02-DR2fit', width=100)

rwin_mode_cplot_select   = pn.widgets.Select(name='Carpet Roll-Win Function',options=['mean','median','min','max','count'], width=200)
rwin_dur_cplot_select    = pn.widgets.IntInput(name='Carpet Roll-Win Duration',start=0,end=10,step=1,value=0, width=200)
sch_only_positives_check = pn.widgets.Checkbox(name='Remove neg. values prior to avg')
sch_only_negatives_check = pn.widgets.Checkbox(name='Remove pos. values prior to avg')
value_cap_input          = pn.widgets.FloatInput(name='Cap value', value=10., width=200, description='Cap value prior to averaging. Entires with absolute values above this value will be capped to the value (keeping the correct sign)')

top_roi_task_select     = pn.widgets.Select(name='ROI Selection (top 50)',options=['All ROIs']+TASKS, width=200, description='ROI Selection for Carpet plot. All ROIs or TOP 50 active ROIs for a given task')

tseg_select             = pn.widgets.Select(name='Scan Segment Selection',options=['All'] + TASKS, width=200, description='Select the scan segment you want to be used for computing the ROI-RS scatter on the right')

schedule_selection_card = pn.Card(schedule_select,pn.Row(denoising_select, model_select), pn.Row(criteria_select, trace_select),title='Schedule Selection', width=250)
schedule_avg_card       = pn.Card(sch_only_positives_check, sch_only_negatives_check, pn.layout.Divider(), rwin_mode_cplot_select, rwin_dur_cplot_select, pn.layout.Divider(), value_cap_input, title='Schedule Avg. Configuration', width=250)
cplot_config_card       = pn.Card(top_roi_task_select,pn.layout.Divider(), cplot_min, cplot_max,title='CPlot Configuration', width=250)
rssts_config_card       = pn.Card(tseg_select, title='ROI-TS Configuration', width=250)

sch_control_col = pn.Column(pn.layout.Divider(),schedule_selection_card,pn.layout.Divider(),schedule_avg_card,pn.layout.Divider(),cplot_config_card, pn.layout.Divider(),rssts_config_card,pn.layout.Divider())


# + editable=true slideshow={"slide_type": ""}
@pn.depends(schedule_select, denoising_select, model_select, criteria_select, trace_select, top_roi_task_select, 
            rwin_mode_cplot_select,rwin_dur_cplot_select, sch_only_positives_check, sch_only_negatives_check, value_cap_input,
            cplot_min, cplot_max, tseg_select)
def sch_options_to_plot(schedule, denoising, model,criteria, trace, task, 
                        rwin_mode,rwin_dur, only_positives, only_negatives, values_cap,
                        vmin, vmax, time_segment):
    
    # Create Schedule carpet plot
    sch_avg_data = _get_roits_schedule_avg(mepfm_ts_xrs.sel(dt=trace), schedule, denoising, model=model, criteria=criteria, 
                            only_negatives=only_negatives, only_positives=only_positives, 
                            rwin_dur=rwin_dur, rwin_mode=rwin_mode,
                            values_cap = values_cap) 
    
    # Select ROIs (if required)
    if task == 'All ROIs':
        data = sch_avg_data
    else:
        data = sch_avg_data.sel(roi=list(roi_info.sort_values(task,ascending=False)['ROI_Name'])[0:50])

    out = create_roits_figure(data,available_traces_types[trace],2.0,'tr','roi',
                      width=2000, height=500, cmap=cplot_colormaps[available_traces_types[trace]], 
                      vmin=vmin, vmax=vmax, 
                      rssts_min=None, rssts_max=None, 
                      roits_min=None, roits_max=None,
                      time_segment=time_segment, hrf_thr=0.2)
    
    return out


# + editable=true slideshow={"slide_type": ""} panel-layout={"width": 100, "height": 10, "visible": true}
dashboard2 = pn.Row(sch_control_col,sch_options_to_plot)

# + editable=true slideshow={"slide_type": ""}
dashboard2
# -



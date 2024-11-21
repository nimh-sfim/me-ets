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

# ***
# # Description
#
# This notebook loads ROI timeseries for E01, E02, and E03 following both basic pre-processing and MEICA denoising.
#
# It then creates two dashboards to explore them as carpet plots. 
#
# Dashboard one allows the exploration of data at the individual scan level
#
# Dashboard two allows the exploration of data at the group level, as a function of task schedule.

# + panel-layout={"width": 100, "height": 0, "visible": true}
import sys
sys.path.append('../python/')

# +
from utils.basics import PRCS_DATA_DIR, SCHEDULES_DIR, DATASET, get_paradigm_info, get_roi_info, TASKS
from utils.basics import ATLAS_DIR_17NETWORKS as ATLAS_DIR
from utils.basics import ATLAS_NAME_17NETWORKS as ATLAS_NAME

from utils.plotting import create_roits_figure, _get_roits_schedule_avg

# + panel-layout={"width": 100, "height": 0, "visible": true}
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

# + panel-layout={"width": 100, "height": 170.114, "visible": true}
roi_info.head()

# + [markdown] panel-layout={"width": 100, "height": 135.241, "visible": true}
# ***
# # 3. Load ROI Level Data
#
# ## 3.1. All Inputs to ME-PFM
#
# Load ROI Timeseries for E01, E02, E03 & E01-MEICA, E02-MEICA and E03-MEICA
# -

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

# + [markdown] panel-layout={"width": 100, "height": 68.4091, "visible": true}
# ***
# # 4. Plot scan-level ROI Timeseries (Basic and MEICA denoised | Separate echoes)

# + editable=true slideshow={"slide_type": ""}
sbj_run_select       = pn.widgets.NestedSelect(options=DATASET, layout=pn.Row, levels=[{'name':'Subject','width':100},{'name':'Run','width':100}])
denoising_select     = pn.widgets.Select(name='SPFM Input',options=['MEICA','Basic'], width=100)
echo_select          = pn.widgets.Select(name='Echo',options=['E01','E02','E03'],value='E02', width=100)
top_roi_task_select  = pn.widgets.Select(name='ROI Selection (top 50)',options=['All ROIs']+TASKS, width=200, description='ROI Selection for Carpet plot. All ROIs or TOP 50 active ROIs for a given task')

only_positives_check = pn.widgets.Checkbox(name='Show only positive values')
only_negatives_check = pn.widgets.Checkbox(name='Show only negative values')

tseg_select          = pn.widgets.Select(name='Scan Segment Selection',options=['All'] + TASKS, width=200, description='Select the scan segment you want to be used for computing the ROI-RS scatter on the right')

scan_selection_card = pn.Card(sbj_run_select,pn.Row(denoising_select, echo_select),title='Scan Selection', width=250)
cplot_config_card   = pn.Card(top_roi_task_select,pn.layout.Divider(), only_positives_check,only_negatives_check,title='CPlot Configuration', width=250)
rssts_config_card   = pn.Card(tseg_select, title='ROI-TS Configuration', width=250)

control_col = pn.Column(pn.layout.Divider(),scan_selection_card,pn.layout.Divider(),cplot_config_card,pn.layout.Divider(),rssts_config_card,pn.layout.Divider())


# + editable=true slideshow={"slide_type": ""}
@pn.depends(sbj_run_select, denoising_select, echo_select, top_roi_task_select, 
            only_positives_check, only_negatives_check,
            tseg_select)
def options_to_plot(sbj_run, denoising, echo, task, only_positives, only_negatives, time_segment):
    sbj = sbj_run['Subject']
    run = sbj_run['Run']
    if task == 'All ROIs':
        data = roi_ts_xrs[sbj,run,denoising].sel(dt=echo)
    else:
        rois = list(roi_info.sort_values(task,ascending=False)['ROI_Name'])[0:50]
        data = roi_ts_xrs[sbj,run,denoising].sel(dt=echo, roi=rois)
    
    aux_attrs = data.attrs
    if only_positives:
        data = xr.where(data > 0, data, np.nan)
    if only_negatives:
        data = xr.where(data < 0, data, np.nan)
    data.attrs = aux_attrs
    
    out = create_roits_figure(data,'BOLD',2.0,'tr','roi',
                      width=2000, height=500, cmap='RdBu_r', 
                      vmin=-0.02, vmax=0.02, 
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
echo_select          = pn.widgets.Select(name='Echo',options=['E01','E02','E03'],value='E02', width=100)

top_roi_task_select      = pn.widgets.Select(name='ROI Selection (top 50)',options=['All ROIs']+TASKS, width=200, description='ROI Selection for Carpet plot. All ROIs or TOP 50 active ROIs for a given task')
rwin_mode_cplot_select   = pn.widgets.Select(name='Carpet Roll-Win Function',options=['mean','median','min','max','count'], width=200)
rwin_dur_cplot_select    = pn.widgets.IntInput(name='Carpet Roll-Win Duration',start=0,end=10,step=1,value=0, width=200)
sch_only_positives_check = pn.widgets.Checkbox(name='Remove neg. values prior to avg')
sch_only_negatives_check = pn.widgets.Checkbox(name='Remove pos. values prior to avg')
value_cap_input          = pn.widgets.FloatInput(name='Cap value', value=10., width=200, description='Cap value prior to averaging. Entires with absolute values above this value will be capped to the value (keeping the correct sign)')

tseg_select             = pn.widgets.Select(name='Scan Segment Selection',options=['All'] + TASKS, width=200, description='Select the scan segment you want to be used for computing the ROI-RS scatter on the right')

schedule_selection_card = pn.Card(schedule_select,pn.Row( denoising_select, echo_select),title='Schedule Selection', width=250)
schedule_avg_card       = pn.Card(top_roi_task_select, pn.layout.Divider(), sch_only_positives_check, sch_only_negatives_check, pn.layout.Divider(), rwin_mode_cplot_select, rwin_dur_cplot_select, pn.layout.Divider(), value_cap_input, title='Schedule Avg. Configuration', width=250)
rssts_config_card       = pn.Card(tseg_select, title='ROI-TS Configuration', width=250)

sch_control_col = pn.Column(pn.layout.Divider(),schedule_selection_card,pn.layout.Divider(),schedule_avg_card,pn.layout.Divider(),rssts_config_card,pn.layout.Divider())


# + editable=true slideshow={"slide_type": ""}
@pn.depends(schedule_select, denoising_select, echo_select, top_roi_task_select, 
            rwin_mode_cplot_select,rwin_dur_cplot_select, sch_only_positives_check, sch_only_negatives_check, value_cap_input,
            tseg_select)
def sch_options_to_plot(schedule, denoising, echo, task, 
                        rwin_mode,rwin_dur, only_positives, only_negatives, values_cap,
                        time_segment):
    # Create Schedule carpet plot
    sch_avg_data = _get_roits_schedule_avg(roi_ts_xrs.sel(dt=echo), schedule, denoising, model=None, criteria=None, 
                            only_negatives=only_negatives, only_positives=only_positives, 
                            rwin_dur=rwin_dur, rwin_mode=rwin_mode,
                            values_cap = values_cap) 
    
    # Select ROIs (if required)
    if task == 'All ROIs':
        data = sch_avg_data
    else:
        data = sch_avg_data.sel(roi=list(roi_info.sort_values(task,ascending=False)['ROI_Name'])[0:50])

    out = create_roits_figure(data,'BOLD',2.0,'tr','roi',
                      width=2000, height=500, cmap='RdBu_r', 
                      vmin=-0.02, vmax=0.02, 
                      rssts_min=None, rssts_max=None, 
                      roits_min=None, roits_max=None,
                      time_segment=time_segment, hrf_thr=0.2)
    
    return out


# + editable=true slideshow={"slide_type": ""} panel-layout={"width": 100, "height": 10, "visible": true}
dashboard2 = pn.Row(sch_control_col,sch_options_to_plot)

# + editable=true slideshow={"slide_type": ""}
dashboard2
# + editable=true slideshow={"slide_type": ""}



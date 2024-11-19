import os.path as osp
import pandas as pd
import numpy as np
ROOT_DIR='/Users/javiergc/bcbl2024'

PRCS_DATA_DIR = osp.join(ROOT_DIR,'PRJ_ME-ETS','BCBL2024','prcs_data')
SCHEDULES_DIR = osp.join(ROOT_DIR,'PRJ_ME-ETS','BCBL2024','code','me-ets','schedules')

ATLAS_NAME_7NETWORKS  ='Schaefer2018_400Parcels_7Networks'
ATLAS_NAME_17NETWORKS ='Schaefer2018_400Parcels_17Networks'

ATLAS_DIR_7NETWORKS   = osp.join(ROOT_DIR,'PRJ_ME-ETS','BCBL2024','atlases',ATLAS_NAME_7NETWORKS)
ATLAS_DIR_17NETWORKS  = osp.join(ROOT_DIR,'PRJ_ME-ETS','BCBL2024','atlases',ATLAS_NAME_17NETWORKS)


ECHOES = {'E01':16.3,'E02':32.2,'E03':48.1}
Nacqs  = 215
DATASET = {'P3SBJ01':['Events01','Events02'],
           'P3SBJ02':['Events01','Events02'],
           'P3SBJ03':['Events01','Events02'],
           'P3SBJ04':['Events01','Events02'],
           'P3SBJ05':['Events01'],
           'P3SBJ06':['Events01'],
           'P3SBJ07':['Events01','Events02'],
           'P3SBJ08':['Events01'],
           'P3SBJ09':['Events01','Events02'],
           'P3SBJ10':['Events01']}

TASKS         = ['FTAP', 'HOUS', 'BMOT', 'READ', 'MUSI']
TASK_COLORS   = {'NoTask':'white','FTAP':'red','HOUS':'green','BMOT':'blue','READ':'yellow','MUSI':'magenta'}
hand2taplabel = {'Right':'RTAP','Left':'LTAP'}

def get_roi_info(ATLAS_DIR, ATLAS_NAME):
    """ Load information about the ROIs in a given atlas

    Inputs
    ======
    ATLAS_DIR: Location of the Atlas
    ATLAS_NAME: Name of the Atlas

    Returns
    =======
    roi_info: pd.DataFrame with ROIs name, hemisphere, network, ID, centroid coordinates and color.
    roi_names: list of all available ROIs in the atlas
    """
    
    roi_info_path = osp.join(ATLAS_DIR,f'{ATLAS_NAME}.roi_info.csv')
    roi_info      = pd.read_csv(roi_info_path)
    roi_names     = list(roi_info['ROI_Name'].values)
    print('++ INFO: Loading information for atlas [%s] with %d ROIs' % (ATLAS_NAME,len(roi_names)))
    return roi_info,roi_names


def get_paradigm_info(DATASET):
    """ Returns information about the task configurations used in every available scan

    Inputs
    ======
    DATASET: dictionary with keys indicating subjects, and values indicating available scans per subject.

    Returns
    =======
    scan2schedule_dict: information about which schedule was used in each run.

    scan2hand_dict: information about what hand was used in the finger tapping task of each scan.

    schedule2evonsets_dict: information about the onsets of all tasks in each scan

    segment_infos: dataframe with info about color and task in each TR.
    """
    scan2schedule_dict     = {}
    scan2hand_dict         = {}
    schedule2evonsets_dict = {}
    schedule_paths         = {}
    segment_infos          = {} 
    for sbj in DATASET.keys():
        for run in DATASET[sbj]:
            paradigm_info_file = osp.join(PRCS_DATA_DIR,sbj,'D00_OriginalData',f'{sbj}_{run}_Paradigm.txt')
            paradigm_info = np.loadtxt(paradigm_info_file,dtype=str)
            schedule = paradigm_info[0]
            scan2schedule_dict[(sbj,run)] = schedule
            hand = paradigm_info[1]
            tap_label = hand2taplabel[hand]
            scan2hand_dict[(sbj,run)] = hand2taplabel[hand]
            schedule_paths[(sbj,run,'FTAP')]=osp.join(SCHEDULES_DIR,f'model_timing.{tap_label}.{schedule}.1D')
            schedule_paths[(sbj,run,'HOUS')]=osp.join(SCHEDULES_DIR,f'model_timing.HOUS.{schedule}.1D')
            schedule_paths[(sbj,run,'BMOT')]=osp.join(SCHEDULES_DIR,f'model_timing.BMOT.{schedule}.1D')
            schedule_paths[(sbj,run,'READ')]=osp.join(SCHEDULES_DIR,f'model_timing.READ.{schedule}.1D')
            schedule_paths[(sbj,run,'MUSI')]=osp.join(SCHEDULES_DIR,f'model_timing.MUSI.{schedule}.1D')
    
            segment_info = pd.DataFrame(index=np.arange(215),columns=['Label','Color'])
            segment_info['Label'] = 'NoTask'
            segment_info['Color'] = 'white'
            for task in ['FTAP','HOUS','BMOT','READ','MUSI']:
                aux_task_onsets = np.loadtxt(schedule_paths[(sbj,run,task)]).astype(int)
                schedule2evonsets_dict[(schedule,task)] = aux_task_onsets
                for i in np.arange(len(aux_task_onsets)):
                    segment_info.loc[aux_task_onsets[0],'Label']   = task
                    segment_info.loc[aux_task_onsets[0]+1,'Label'] = task
                    segment_info.loc[aux_task_onsets[0]+2,'Label'] = task
                    segment_info.loc[aux_task_onsets[0],'Color']   = TASK_COLORS[task]
                    segment_info.loc[aux_task_onsets[0]+1,'Color'] = TASK_COLORS[task]
                    segment_info.loc[aux_task_onsets[0]+2,'Color'] = TASK_COLORS[task]
            segment_infos[(sbj,run)] = segment_info
    return scan2schedule_dict, scan2hand_dict, schedule2evonsets_dict, segment_infos
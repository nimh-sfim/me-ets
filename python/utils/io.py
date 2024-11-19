import xarray as xr
import os.path as osp
from tqdm import tqdm
from .basics import PRCS_DATA_DIR, get_paradigm_info


def load_orig_roits_xr(DATASET,ATLAS_NAME):
    scan2schedule_dict, scan2hand_dict, schedule2evonsets_dict, _ = get_paradigm_info(DATASET)
    ds = xr.Dataset()
    for sbj in tqdm(DATASET.keys(),desc='Subject'):
        for run in DATASET[sbj]:
            scan_name = f'{sbj}_{run}'
            data_path = osp.join(PRCS_DATA_DIR,sbj,'D04_ROIts',f'{sbj}_{run}_{ATLAS_NAME}.roits.nc')
        
            ds[scan_name]                   = xr.open_dataarray(data_path)
            ds[scan_name].attrs['sbj']      = sbj
            ds[scan_name].attrs['run']      = run
            ds[scan_name].attrs['schedule'] = scan2schedule_dict[sbj,run]

def load_mepfm_roits_xr(DATASET,ATLAS_NAME):
    scan2schedule_dict, scan2hand_dict, schedule2evonsets_dict, _ = get_paradigm_info(DATASET)
    ds = xr.Dataset()
    for sbj in tqdm(DATASET.keys(),desc='Subject'):
    for run in DATASET[sbj]:
        for denoising in ['Basic','MEICA']:
            for criteria in ['bic','aic']:
                for model in ['R2ONLY','RHO_ZERO']:
                    data_path = osp.join(PRCS_DATA_DIR,sbj,'D06_MEPFM',f'{sbj}_{run}_MEPF.{denoising}_{criteria}_{model}.{ATLAS_NAME}.nc')
                    if not osp.exists(data_path):
                        output_xrs[(sbj,run,denoising,criteria,model)] = None
                        print("++ WARNING: Missing file [%s]" % data_path)
                        continue
                    else:
                        output_xrs[(sbj,run,denoising,criteria,model)] = xr.open_dataarray(data_path)
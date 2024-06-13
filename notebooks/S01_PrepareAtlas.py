# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: ME-ETS (2024a | BCBL)
#     language: python
#     name: me_ets_2024a
# ---

# # Description
#
# We decided we will be working with the version of the data from NI2019 that Cesar has at BCBL. We will start our analysis with the multi-echo data in MNI space that has already undergone MEICA denoising (e.g., pc08 files).
#
# #### 1. The first I did was to make sure that all pc08 files (all subjects, all echoes, all runs) are in the same grid
#
# ```bash
# # cd /data/SFIMJGC/PRJ_ME-ETS/BCBL2024/prcs_data/
# 3dinfo -same_all_grid ./P3SBJ??/D02_Preprocessed/pc08.P3SBJ??_Events0?_MEICA.E0?.spc.nii.gz
# ```
#
# This returned all 1s (so yes, all in same grid)
#
# ***

# #### 2. Download the Schaefer Atlas 400ROI/7NW from Thomas Yeo's github into the atlas folder, and then bring it into the pc08 grid
#
# ```bash
# # cd ${ATLAS_DIR}
# 3drefit -space MNI Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz
#
# @MakeLabelTable -lab_file Schaefer2018_400Parcels_7Network_order.txt 1 0 \
#                 -labeltable Schaefer2018_400Parcels_7Network.niml.lt \
#                 -dset Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz 
#
# 3dresample -inset Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz \
#            -master ../prcs_data/P3SBJ06/D02_Preprocessed/pc08.P3SBJ06_Events01_MEICA.E01.spc.nii.gz \
#            -rmode NN \
#            -prefix Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.pc08grid.nii.gz
#
# 3drefit -labeltable Schaefer2018_400Parcels_7Networks.niml.lt Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.pc08grid.nii.gz
# ```
# ***

# #### 3. Use 3dNetCorr to find the FOV common to all Sbjs/Runs/Echoes (according to 3dNetCorr)
#
# ```bash
# sh /data/SFIMJGC/PRJ_ME-ETS/BCBL2024/code/me-ets/bash/check_atlas_vs_fov.sh
# ```
#
# ***

# #### 4. Get a list with the ROIs that we need to remove from the altas in order to compute FC / edges with the same ROIs in all Sbj/Run/Echoes

import pandas as pd
from glob import glob
import os.path as osp
import subprocess
PRCS_DATA_DIR = '/data/SFIMJGC/PRJ_ME-ETS/BCBL2024/prcs_data/'
ATLAS_DIR='/data/SFIMJGC/PRJ_ME-ETS/BCBL2024/atlases/Schaefer2018_400Parcels_7Networks/'
ATLAS_NAME='Schaefer2018_400Parcels_7Networks'

roidat_files = glob(PRCS_DATA_DIR+'/*/*/rm.*_FOVcheck_000.roidat')
bad_roi_list = []
for roidata_path in roidat_files:
    roidata_file = osp.basename(roidata_path)
    sbj = (roidata_file.split('.')[1]).split('_')[0]
    run = (roidata_file.split('.')[1]).split('_')[1]
    echo = (roidata_file.split('.')[2]).split('_')[0]
    roidat_df = pd.read_csv(roidata_path,sep=' ', skipinitialspace=True, header=0)
    correct_columns = roidat_df.columns.drop(['#'])
    roidat_df = roidat_df.drop(['ROI_label'],axis=1)
    roidat_df.columns = correct_columns
    roidat_df = roidat_df.drop(['#.1'],axis=1)
    bad_rois = roidat_df[roidat_df['frac']<=0.1][['ROI','ROI_label']]
    print('++ INFO: %s/%s/%s --> Number of Bad Rois: %d' % (sbj,run,echo,bad_rois.shape[0]), end=' | ')
    for i,br in bad_rois.iterrows():
        bad_roi_list.append((br['ROI'],br['ROI_label']))

bad_roi_list = list(set(bad_roi_list))

print(bad_roi_list)

print('++ INFO: Number of ROIs to remove = %d ROIs' % len(bad_roi_list))

# ***
#
# #### 5. Remove regions with bad coverage from atlas

bad_rois_minus = '-'.join([str(r)+'*equals(a,'+str(r)+')' for r,rs in bad_roi_list])
bad_rois_plus  = '+'.join([str(r)+'*equals(a,'+str(r)+')' for r,rs in bad_roi_list])
print(bad_rois_minus)
print(bad_rois_plus)

command=f"""module load afni; \
           cd {ATLAS_DIR}; \
           3dcalc -overwrite \
                  -a {ATLAS_NAME}_order_FSLMNI152_2mm.pc08grid.nii.gz \
                  -expr '{bad_rois_plus}' \
                  -prefix {ATLAS_NAME}_order_FSLMNI152_2mm.pc08grid.RemovedROIs.nii.gz; \
           3dcalc -overwrite \
                  -a      {ATLAS_NAME}_order_FSLMNI152_2mm.pc08grid.nii.gz \
                  -expr 'a-{bad_rois_minus}' \
                  -prefix rm.{ATLAS_NAME}_order_FSLMNI152_2mm.pc08grid.fov_restricted.nii.gz; \
           3drefit -labeltable {ATLAS_NAME}.niml.lt rm.{ATLAS_NAME}_order_FSLMNI152_2mm.pc08grid.fov_restricted.nii.gz"""
output  = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
print(output.strip().decode())

from nilearn.plotting import plot_roi

plot_roi(osp.join(ATLAS_DIR,f'{ATLAS_NAME}_order_FSLMNI152_2mm.pc08grid.RemovedROIs.nii.gz'),title='ROIs that will be removed from the ATLAS')

plot_roi(osp.join(ATLAS_DIR,f'{ATLAS_NAME}_order_FSLMNI152_2mm.pc08grid.nii.gz'),title='Original ATLAS')

plot_roi(osp.join(ATLAS_DIR,f'rm.{ATLAS_NAME}_order_FSLMNI152_2mm.pc08grid.fov_restricted.nii.gz'),title='FOV-Restricted ATLAS')

# ***
# #### 6. Rank the FOV-restricted Atlas

command = f"""module load afni; \
             cd {ATLAS_DIR}; \
             3dRank -prefix rm.{ATLAS_NAME}_order_FSLMNI152_2mm.pc08grid.fov_restricted.ranked.nii.gz -input rm.{ATLAS_NAME}_order_FSLMNI152_2mm.pc08grid.fov_restricted.nii.gz;"""
output  = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
print(output.strip().decode())

# ***
# #### 7. Create rank corrected Order & Centroid Files

from sfim_lib.atlases.raking import correct_ranked_atlas

path_to_order_file     = osp.join(ATLAS_DIR,f'{ATLAS_NAME}_order.txt')
path_to_rank_file      = osp.join(ATLAS_DIR,f'rm.{ATLAS_NAME}_order_FSLMNI152_2mm.pc08grid.fov_restricted.ranked.nii.gz.rankmap.1D')
path_to_centroids_file = osp.join(ATLAS_DIR,f'{ATLAS_NAME}_order_FSLMNI152_2mm.Centroid_RAS.csv')
correct_ranked_atlas(path_to_order_file,path_to_centroids_file,path_to_rank_file)

# ***
# #### 8. Add corrected label table to the ranked version of the atlas

command = f"""module load afni; \
             cd {ATLAS_DIR}; \
             @MakeLabelTable -lab_file {ATLAS_NAME}_order.ranked.txt 1 0 -labeltable {ATLAS_NAME}_order.ranked.niml.lt -dset rm.{ATLAS_NAME}_order_FSLMNI152_2mm.pc08grid.fov_restricted.ranked.nii.gz;"""
output  = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
print(output.strip().decode())


# ***
# #### 9. Create a Dataframe with all the necessary info about the final FOV-restricted Atlas

def rgb2hex(r,g,b):
    """
    Converts between two different versions of RGB color codes. Input as three separate integers between 0 and 256.
    Output will be in hexadecimal code.
    """
    return "#{:02x}{:02x}{:02x}".format(r,g,b)


# Load the cetroid file for the ranked atlas in memory
centroids_info               = pd.read_csv(osp.join(ATLAS_DIR,f'{ATLAS_NAME}.Centroid_RAS.ranked.csv' ))
centroids_info['ROI Name']   = [label.split('7Networks_')[1] for label in centroids_info['ROI Name']]
centroids_info['Hemisphere'] = [item.split('_')[0] for item in centroids_info['ROI Name']]
centroids_info['Network']    = [item.split('_')[1] for item in centroids_info['ROI Name']]
# Load the color info file for the ranked atlas in memory
color_info = pd.read_csv(osp.join(ATLAS_DIR, f'{ATLAS_NAME}_order.ranked.txt'),sep='\t', header=None)
# Combine all the useful columns into a single new dataframe
df         = pd.concat([centroids_info[['ROI Label','Hemisphere','Network','ROI Name','R','A','S']],color_info[[2,3,4]]], axis=1)
df.columns = ['ROI_ID','Hemisphere','Network','ROI_Name','pos_R','pos_A','pos_S','color_R','color_G','color_B']
df['RGB']  = [rgb2hex(r,g,b) for r,g,b in df.set_index('ROI_ID')[['color_R','color_G','color_B']].values]
# Save the new data frame to disk
df.to_csv(osp.join(ATLAS_DIR,f'{ATLAS_NAME}.roi_info.csv'), index=False)

df.head(5)

# ***
# #### 10. Clean-up atlas directory

# ```bash
# # cd /data/SFIMJGC/PRJ_ME-ETS/BCBL2024/atlases/Schaefer2018_400Parcels_7Networks/
#
# # mkdir orig
# # mv Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv orig
# # mv Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz orig
# # mv Schaefer2018_400Parcels_7Networks_order.lut orig
# # mv Schaefer2018_400Parcels_7Networks_order.txt orig
# # mv Schaefer2018_400Parcels_7Networks.niml.lt orig
#
# # mkdir pc08grid
# # mv Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.pc08grid.nii.gz pc08grid
# # mv Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.pc08grid.RemovedROIs.nii.gz pc08grid
# # mv rm.Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.pc08grid.fov_restricted.nii.gz pc08grid
#
# # mv rm.Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.pc08grid.fov_restricted.ranked.nii.gz Schaefer2018_400Parcels_7Networks.nii.gz
# # mv Schaefer2018_400Parcels_7Networks_order.ranked.niml.lt Schaefer2018_400Parcels_7Networks.niml.lt
# # mv Schaefer2018_400Parcels_7Networks.Centroid_RAS.ranked.csv Schaefer2018_400Parcels_7Networks.csv
# # mv Schaefer2018_400Parcels_7Networks_order.ranked.txt Schaefer2018_400Parcels_7Networks_order.txt
# # mv rm.Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.pc08grid.fov_restricted.ranked.nii.gz.rankmap.1D Schaefer2018_400Parcels_7Networks.rankmap.1D
# ```

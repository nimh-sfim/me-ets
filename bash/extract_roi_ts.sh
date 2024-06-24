# =============================================================
# Author: Javier Gonzalez-Castillo
# Date: June 13th, 2014 (@BCBL)
# Goal: Get FOV mask (as 3dNetCorr)
# =============================================================
PRCS_DATA_DIR=/data/SFIMJGC/PRJ_ME-ETS/BCBL2024/prcs_data
ATLAS_NAME=Schaefer2018_400Parcels_7Networks
ATLAS_PATH=`echo /data/SFIMJGC/PRJ_ME-ETS/BCBL2024/atlases/${ATLAS_NAME}/${ATLAS_NAME}.nii.gz`
set -e
ml afni

cd ${PRCS_DATA_DIR}
Sbj_folders=`find -mindepth 1 -maxdepth 1 -type d`
echo ${Sbj_folders}

for folder in ${Sbj_folders}
do
	sbj=`basename ${folder}`
  echo " ----------------------------------------> ${sbj} <-------------------------------------------"
	cd ${PRCS_DATA_DIR}/${folder}/D02_Preprocessed/
  for run in Events01 Events02
  do
		for echo in E01 E02 E03
		do
			DATA_PATH=`echo pc08.${sbj}_${run}_MEICA.${echo}.spc.nii.gz`
			if [ -f ${DATA_PATH} ]; then
				3dNetCorr -overwrite \
            -mask rm.${sbj}_${run}_MEICA.${echo}_FOVcheck_mask_nnull+tlrc \
            -in_rois ${ATLAS_PATH} \
            -inset ${DATA_PATH} \
            -ts_out \
						-prefix ${sbj}_${run}_MEICA-${echo}.${ATLAS_NAME}
			else
				echo "++ WARNING: [${DATA_PATH}] missing. Nothing happened for this Sbj/Run/Echo combination."
			fi
		done
	done
done

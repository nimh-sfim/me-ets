# =============================================================
# Author: Javier Gonzalez-Castillo
# Date: June 13th, 2014 (@BCBL)
# Goal: Get FOV mask (as 3dNetCorr)
# =============================================================
PRCS_DATA_DIR=/data/SFIMJGC/PRJ_ME-ETS/BCBL2024/prcs_data
ATLAS_PATH=/data/SFIMJGC/PRJ_ME-ETS/BCBL2024/atlases/Schaefer2018_400Parcels_7Networks/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.pc08grid.nii.gz
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
            -in_rois ${ATLAS_PATH} \
            -output_mask_nonnull \
            -inset ${DATA_PATH} \
						-prefix rm.${sbj}_${run}_MEICA.${echo}_FOVcheck
			else
				echo "++ WARNING: [${DATA_PATH}] missing. Nothing happened for this Sbj/Run/Echo combination."
			fi
		done
	done
done

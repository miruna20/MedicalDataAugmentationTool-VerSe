#!/bin/bash

export PYTHONPATH=/MedicalDataAugmentationTool

mkdir /MedicalDataAugmentationTool-VerSe/verse2019_dataset/images/output

#reorient all
python /MedicalDataAugmentationTool-VerSe/other/reorient_reference_to_rai.py --image_folder /MedicalDataAugmentationTool-VerSe/verse2019_dataset/images  --output_folder /MedicalDataAugmentationTool-VerSe/verse2019_dataset/images_reoriented

#run training scripts
cd /MedicalDataAugmentationTool-VerSe/training
mkdir /output
python main_spine_localization.py 
python main_vertebrae_localization.py
python main_vertebrae_segmentation.py

#copy the results 
cp MedicalDataAugmentationTool-VerSe/training/output/* /MedicalDataAugmentationTool-VerSe/verse2019_dataset/images/output/

# use tensorflow-gpu base image
FROM tensorflow/tensorflow:1.15.0-gpu-py3 as base
# install python dependencies
RUN pip3 install SimpleITK itk scikit-image cachetools Pyro4
ADD MedicalDataAugmentationTool /MedicalDataAugmentationTool
# copy verse files
ADD MedicalDataAugmentationTool-VerSe/training /MedicalDataAugmentationTool-VerSe/training
ADD MedicalDataAugmentationTool-VerSe/other /MedicalDataAugmentationTool-VerSe/other
ADD MedicalDataAugmentationTool-VerSe/*.py /MedicalDataAugmentationTool-VerSe/
ADD MedicalDataAugmentationTool-VerSe/verse2019_dataset /MedicalDataAugmentationTool-VerSe/verse2019_dataset
ADD MedicalDataAugmentationTool-VerSe/docker/train.sh /

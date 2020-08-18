import argparse
import os
import pandas as pd
import nibabel as nib
import numpy as np

#in order to compute dice scores, padding is needed for cropped images

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--reorientedImagesFolder", help="folder with the rai oriented images",required=True)
    parser = parser.parse_args()
    print(parser.reorientedImagesFolder)

    imagesFolder = parser.reorientedImagesFolder

    cropping_params = pd.read_csv(os.path.join(imagesFolder,"cropping_params.csv"), sep=';', header=None)

    for index, row in cropping_params.itertuples():
        splitted = row.split(',')
        name_image = splitted[0]
        cropping_param = int(splitted[1])
        image = nib.load(os.path.join(imagesFolder,name_image + ".nii.gz"))
        img_numpyarray = np.array(image.dataobj)
        shape_pic = np.shape(img_numpyarray)
        padded_array = np.zeros((shape_pic[0], shape_pic[1], shape_pic[2] + cropping_param))
        shape_pad = np.shape(padded_array)
        shape0pad = np.shape(padded_array[0])
        shape1pad = np.shape(padded_array[1])
        shape2pad = np.shape(padded_array[2])
        shape0img = np.shape(img_numpyarray[0])
        shape1img = np.shape(img_numpyarray[1])
        shape2img = np.shape(img_numpyarray[2])

        padded_array[0:shape_pic[0], 0:shape_pic[1],cropping_param:(shape_pic[2] + cropping_param)] = img_numpyarray

        new_image = nib.Nifti1Image(padded_array, image.affine)
        nib.save(new_image, os.path.join(imagesFolder,"test",name_image + ".nii.gz"))



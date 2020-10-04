import json
import os
from glob import glob

import numpy as np
from utils.io.image import read_meta_data
from utils.io.landmark import save_points_csv
from utils.io.text import save_dict_csv
from utils.landmark.common import Landmark


def save_valid_landmarks_list(landmarks_dict, filename):
    """
    Saves the valid landmarks per image id to a file.
    :param landmarks_dict: A dictionary of valid landmarks per image id.
    :param filename: The filename to where to save.
    """
    valid_landmarks = {}
    for image_id, landmarks in landmarks_dict.items():
        current_valid_landmarks = []
        for landmark_id, landmark in enumerate(landmarks):
            if landmark.is_valid:
                current_valid_landmarks.append(landmark_id)
        valid_landmarks[image_id] = current_valid_landmarks
    save_dict_csv(valid_landmarks, filename)


if __name__ == '__main__':
    # this script converts the landmark lml files from the Ben Glocker dataset to
    # a landmark.csv file with physical coordinates

    dataset_folder = '/home/miruna20/Documents/Datasets/SpinewebDatasets/self-supervised_pretraining/all_data'
    landmark_mapping = dict([(i + 1, i) for i in range(25)])
    num_landmarks = len(landmark_mapping)
    landmarks_dict = {}
    files = glob(os.path.join(dataset_folder, '*.lml'))
    for filename in sorted(files):
        # get image id
        filename_wo_folder = os.path.basename(filename)
        filename_wo_folder_and_ext = filename_wo_folder[:-4]
        image_id = filename_wo_folder_and_ext
        print(filename_wo_folder_and_ext)
        # get image meta data
        image_meta_data = read_meta_data(os.path.join(dataset_folder, image_id + '.nii.gz'))
        spacing = np.array(image_meta_data.GetSpacing())
        origin = np.array(image_meta_data.GetOrigin())
        direction = np.array(image_meta_data.GetDirection()).reshape([3, 3])
        size = np.array(image_meta_data.GetSize())
        # placeholder for landmarks
        current_landmarks = [Landmark([np.nan] * 3, False, 1.0, 0.0) for _ in range(num_landmarks)]
        with open(filename, 'r') as f:
            # first read all of the lines from the file
            lines = f.readlines()
        ####Label encoding for Ben Glocker Data#####
        # C1 --> 10, C7 --> 70
        # T1 --> 80, T12 --> 190
        # L1 --> 200, L5 --> 240
        # iterate over the lines and extract the coordinates
        dataset_with_cervical = False
        for line in lines:
            separated = line.split('\t')

            #do not process first line
            if not separated[0].isalnum():
                continue
            vert_id = int(int(separated[0]) / 10)
            # make sure only centroids of the cervical vertebrae are saved
            if vert_id > 7:
                continue
            else:
                dataset_with_cervical = True

            # read coordinates from the ben glocker lml Are they already in the physical coordinates?
            coords = np.array([float(separated[2]),float(separated[3]),float(separated[4])])

            # conversion for verse 2019
            # coords = np.array([float(landmark['Z']), float(landmark['Y']), size[2] * spacing[2] - float(landmark['X'])])

            # labels in verse start at 1, our indexing starts at 0
            index = landmark_mapping[vert_id]
            current_landmarks[index].coords = coords
            current_landmarks[index].is_valid = True
            print(coords)
        if dataset_with_cervical:
            landmarks_dict[image_id] = current_landmarks
        else:
            os.remove(os.path.join(dataset_folder,"images_reoriented",image_id + ".nii.gz"))

    save_points_csv(landmarks_dict, os.path.join(dataset_folder, 'landmarks.csv'))
    save_valid_landmarks_list(landmarks_dict, os.path.join(dataset_folder, 'valid_landmarks.csv'))

__author__ = 'Dhananjay Mehta and Swapnil Kumar'

from glob import glob
import numpy as np
import pandas as pd
from PIL import Image
import os
import cPickle
import gzip

"""
    This program will generate pickle file for Stage 1 of OCR pipeline.
    Generate files that combines the text data and non-text data in follwing ratio -
    70:15:15 for Training:Validation:Testing
"""


def dir_to_dataset(new_path, loc_train_labels=""):
    """
    This function extracts the data from the folder and returns resized array of grayscale image [28:28]
    :param new_path: Location of gray scaled images
    :param loc_train_labels: Training labels for data
    :return:
    """
    dataset = []
    glob_files_updated_path = new_path + "*.png"
    for file_count, file_name in enumerate(sorted(glob(glob_files_updated_path), key=len)):
        img = Image.open(file_name).convert('LA')
        img = img.resize((28, 28))
        pixels = [f[0] for f in list(img.getdata())]
        dataset.append(pixels)
    return dataset


def create_dataset(valid_dir_path, invalid_dir_path):
    """
    This function will generate
    :param valid_dir_path:  directory path for valid images (positive training)
    :param invalid_dir_path: directory path for invalid images (negative training)
    :return:
            None
    """
    final_train_set_x = np.zeros((1, 784))  # final training dataset of images
    final_train_set_y = np.array([])  # final training dataset with the labels
    final_val_set_x = np.zeros((1, 784))  # final validation dataset
    final_val_set_y = np.array([])  # finla validation dataset  with the labels
    final_test_set_x = np.zeros((1, 784))  # final data to test
    final_test_set_y = np.array([])  # final data to test with labels

    # generating the data for the VALID INPUT  in scale - train:validate:test - 70:15:15
    Sample_list = os.listdir(valid_dir_path)
    for dir_in_sample in Sample_list:
        new_path = valid_dir_path + dir_in_sample + "/"
        if (dir_in_sample != ".DS_Store"):
            valid_data_training = dir_to_dataset(new_path)
            leng = len(valid_data_training)
            np_valid_data_training = np.array(valid_data_training)
            np_valid_y_training = np.ones(leng)

            # : train_set_x -  calculate the value of training data - for images
            # : train_set_y -  calculate the value of training data - for labels
            train_set_x = np_valid_data_training[0:.7 * (leng)]
            train_set_y = np_valid_y_training[0:.7 * (leng)]

            # : val_set_x -  calculate the value of validation data - for images
            # : val_set_y -  calculate the value of validation data - for labels
            val_set_x = np_valid_data_training[.7 * (leng): 0.85 * (leng)]
            val_set_y = np_valid_y_training[.7 * (leng): 0.85 * (leng)]

            # : test_set_x -  calculate the value of validation data - for images
            # : test_set_y -  calculate the value of validation data - for labels
            test_set_x = np_valid_data_training[.85 * (leng): 1 * (leng)]
            test_set_y = np_valid_y_training[.85 * (leng): 1 * (leng)]

            # print "shape",train_set_x.shape
            # print "shape",train_set_y.shape
            final_train_set_x = np.concatenate((final_train_set_x, train_set_x), axis=0)
            final_train_set_y = np.concatenate((final_train_set_y, train_set_y), axis=0)
            final_val_set_x = np.concatenate((final_val_set_x, val_set_x), axis=0)
            final_val_set_y = np.concatenate((final_val_set_y, val_set_y), axis=0)
            final_test_set_x = np.concatenate((final_test_set_x, val_set_x), axis=0)
            final_test_set_y = np.concatenate((final_test_set_y, val_set_y), axis=0)

    # generating data for the INVALID INPUT in scale - train:validate:test - 70:15:15
    np_invalid_data_training = np.array(dir_to_dataset(invalid_dir_path))  # fetch the data for invalid dataset
    leng2 = len(np_invalid_data_training)
    np_invalid_y_training = np.zeros(leng2)
    print("final_train_set_x", final_train_set_x.shape)
    print("np_invalid_data_t", np_invalid_data_training.shape)

    final_train_set_x = np.concatenate((final_train_set_x, np_invalid_data_training[0:int(.7 * leng2)]), axis=0)
    final_train_set_y = np.concatenate((final_train_set_y, np_invalid_y_training[0:int(.7 * leng2)]), axis=0)

    final_val_set_x = np.concatenate((final_val_set_x, np_invalid_data_training[int(.7 * leng2): int(.85 * leng2)]),
                                     axis=0)
    final_val_set_y = np.concatenate((final_val_set_y, np_invalid_y_training[int(.7 * leng2): int(.85 * leng2)]),
                                     axis=0)
    final_test_set_x = np.concatenate((final_test_set_x, np_invalid_data_training[int(.85 * leng2): 1 * leng2]), axis=0)
    final_test_set_y = np.concatenate((final_test_set_y, np_invalid_y_training[int(.85 * leng2): 1 * (leng2)]), axis=0)

    # Concatenate the data for the valid dataset (dataset containing images with text)
    # and invalid dataset(dataset containing images without text).

    train_set = final_train_set_x, final_train_set_y
    val_set = final_val_set_x, final_val_set_y
    test_set = final_test_set_x, final_test_set_y

    data_set = [train_set, val_set, test_set]
    print("writing to file")

    f = gzip.open('char_num_check.pkl.gz', 'wb')
    cPickle.dump(data_set, f, protocol=2)
    print("file written")
    f.close()

__author__ = 'Dhananjay Mehta and Swapnil Kumar'
from glob import glob
import numpy as np
from PIL import Image
import pandas as pd
import os
import cPickle
import gzip


def dir_to_dataset(glob_files, loc_train_labels=""):
    dataset = []
    for file_name in enumerate(sorted(glob(glob_files),key=len) ):
        img = Image.open(file_name).convert('LA')
        img = img.resize((28,28))
        pixels = [f[0] for f in list(img.getdata())]
        dataset.append(pixels)
    if len(loc_train_labels) > 0:
        data_field = pd.read_csv(loc_train_labels)
        return dataset, data_field["Classification"].tolist()
    else:
        return dataset


def create_dataset(dir_path, csv_path):
    list_dir = os.listdir(dir_path)
    data_training = []
    y_training = []
    i = 1
    for dir in list_dir:
        file_path = dir_path + dir+ "/*.png"
        csv_file_name = csv_path + dir + ".csv"

        file = open(csv_file_name, 'w')
        file.write("Classification")
        if i <= 10:
            for j in range(len(glob(file_path))):
                file.write("\n"+str(47+i))
        if i >10 and i<=36:
            for j in range(len(glob(file_path))):
                file.write("\n"+str(i+54))
        if i >36:
            for j in range(len(glob(file_path))):
                file.write("\n"+str(i+60))

        i += 1
        file.close()
        data, y = dir_to_dataset(file_path,csv_file_name)
        data_training += data
        y_training += y

    np_data_training = np.array(data_training)
    np_y_training = np.array(y_training)

    for i in range(0,62):
        j = i*(1016)
        if i ==0:
            train_set_x = np_data_training[j:j+816]
            train_set_y = np_y_training[j:j+816]

            val_set_x = np_data_training[j+816:j+916]
            val_set_y = np_y_training[j+816:j+916]

            test_set_x = np_data_training[j+916:j+1015]
            test_set_y = np_y_training[j+916:j+1015]
        else:
            train_set_x = np.append(train_set_x,np_data_training[j:j+816],axis=0)
            train_set_y = np.concatenate((train_set_y,np_y_training[j:j+816]),axis=0)

            val_set_x = np.append(val_set_x,np_data_training[j+816:j+916], axis = 0)
            val_set_y = np.concatenate((val_set_y,np_y_training[j+816:j+916]),axis = 0)

            test_set_x = np.append(test_set_x,np_data_training[j+916:j+1015],axis = 0)
            test_set_y = np.concatenate((test_set_y,np_y_training[j+916:j+1015]),axis = 0)

    train_set = train_set_x, train_set_y
    val_set = val_set_x, val_set_y
    test_set = test_set_x, val_set_y

    data_set = [train_set, val_set, test_set]

    f = gzip.open('char_num.pkl.gz','wb')
    cPickle.dump(data_set, f, protocol=2)
    f.close()

#if __name__ == '__main__':
#    create_dataset(dir_path, csv_path)
#    dir_path = "/Users/dhananjaymehta/CIS730/Vision/EnglishFnt/Fnt/"
#    csv_path = "/Users/dhananjaymehta/CIS730/Vision/EnglishFnt/CSV/"

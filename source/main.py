__author__ = 'Dhananjay Mehta and Swapnil Kumar'

"""
    --------------------
        DESCRIPTION
    --------------------

    CIS730 : Artificial Intelligence
    Text recognition from Unstructured Image and Natural Scenes using Supervised Learning Algorithms.
    Used Deep Learning library 'Theano' to implement the algorithm.

"""

import make_dataset_Stage1
import make_dataset_Stage2
from preprocess_image import ProcessImage
from check_character import CheckCharacter
from predict_character import PredictCharacter
from logistic_regression import sgd_optimization_mnist

if __name__ == '__main__':

    valid_dir_path = "/Users/dhananjaymehta/CIS730/Vision/Training_Dataset/valid_dataset_1/"
    invalid_dir_path = "/Users/dhananjaymehta/CIS730/Vision/Training_Dataset/Invalid_dataset_0/*/"
    csv_path = "/Users/dhananjaymehta/CIS730/Vision/EnglishFnt/CSV2/"

    # The first step will be creation of the dataset for the Stage 1 of the OCR pipeline from -
    # Dataset containing images with text - The Chars74K dataset
    # Dataset containing random images without text - Kaggle CIFAR-10
    make_dataset_Stage1.create_dataset(valid_dir_path, invalid_dir_path)

    # This step will generate the dataset for the Stage 2 of the OCR pipeline from - Chars74K dataset
    make_dataset_Stage2.create_dataset(valid_dir_path, csv_path)

    # This step will train the model for Stage 2 for the OCR pipeline and generate the best model
    sgd_optimization_mnist(dataset='char_num_check.pkl.gz', n_epochs=1000, batch_size= 3000 ,best_model='best_model_char_num_check.pkl', n_output = 2);

    # This step will train the model for Stage 3 for the OCR pipeline and generate the best model
    sgd_optimization_mnist(dataset='char_num_classify.pkl.gz', n_epochs=2000, batch_size= 350, best_model='best_model_char_num_classify.pkl', n_output = 123);

    # Stage 1: Pre process the input image.
    image = ProcessImage('hello_world.jpg')

    # Stage 1: Pre process the input image to get the contour and image objects.
    candidates = image.get_text_candidates()

    # Stage 2: Text Detection, detects objects containing text within the image.
    final_candidates = CheckCharacter().check_character(candidates["flattened"], best_model='best_model_char_num_check.pkl')
    final_candidates = candidates["flattened"]

    # Stage 3: Character Recognition, Determines the character contained in the image.
    PredictCharacter().predict_character(final_candidates, best_model='best_model_char_num_classify.pkl')
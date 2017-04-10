Object Detection using DeepLearning Framework - Theano 
======================================================

# Table of Contents
1. [Problem Summary](README.md#problem-summary)
2. [Data](README.md#data)
3. [Implementation](README.md#details-of-implementation)
4. [Algorithms](README.md#Algorithms)


## Problem Summary
[Back to Table of Contents](README.md#table-of-contents)

This project was part of my course work CIS730(Principles of Artificial Intellegience) in Fall 2015. This project implements supervised learning algorithms using Theano to recognize text from Unstructured Image and Natural Scenes.

## Data
[Back to Table of Contents](README.md#table-of-contents)

Dataset used in the project was used from "Chars74K dataset" and "Kaggle CIFAR-10"

## Implementation
[Back to Table of Contents](README.md#table-of-contents)

This project sets up a three stage pipeline for building an object chatacter recognition. 
#### Stage 1: Pre process the input image and get the contour and image objects.
#### Stage 2: Text Detection, detects objects containing text within the image.
#### Stage 3: Character Recognition, Determine characters contained in the image.

First stage is involved with data preprocessing. This involve cleaning data and labelling the images. The dataset for the Stage 1 of the OCR pipeline is created using following steps : 
1. Dataset containing images with text - The Chars74K dataset
2. Dataset containing random images without text - Kaggle CIFAR-10

Now the next step is to generate the dataset labels for Chars74K dataset. Once the data is created and labelled third step is to train the model on these images. The model trained in this step will be used to classify the images in Stage 2 for the OCR pipeline and generate the best model

## Algorithm
[Back to Table of Contents](README.md#table-of-contents)
Logistic regression was used for building the OCR. Logistic regression was implemented using Theano Library.  


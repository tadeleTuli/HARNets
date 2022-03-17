#!/usr/bin/env python
# coding: utf-8

'''
This module helps to predict new data sets using a trained model
Author: Tadele Belay Tuli, Valay Mukesh Patel 

University of Siegen, Germany (2022)
License: MIT
'''
import glob
import os
import subprocess
import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

import tensorflow 
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Activation, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Import module from training model
import Skeleton_LSTM_Training as slt


def load_model():
    """Load the pre-trained model weights from the current directory"""
    model = load_model('Skeleton_model.h5')
    return model      



def main():

    # Data preparation
    pretrained_model = load_model()

    # Convert Dataset from BVH to CSV
    slt.convert_dataset_to_csv("file_to_predict/*")

    # Load CSV as dataframe
    df_prediction = slt.convert_CSV_into_df("file_to_predict/*")

    # Creating batch with time_steps and features from data frame.
    steps = time_steps = 200

    data = []
    for i in range(0, df_prediction.shape[0] - time_steps, steps):
        z = df_prediction[df_prediction.columns[0:df_prediction.shape[1]]].values[i: i + time_steps]
        data.append(z)

    # Converting list data into numpy array. 
    data_to_predict = np.array(data)   
    print(data_to_predict.shape)

    # Prediction of activity from pretrained model.
    predict = pretrained_model.predict_classes(data_to_predict)   
    print("Prints the one hot encoded label of class (0-8)", predict)                                       


if __name__ == '__main__':
    main()
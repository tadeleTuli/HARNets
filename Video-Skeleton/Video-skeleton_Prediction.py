#!/usr/bin/env python
# coding: utf-8


import glob
import os
import subprocess
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, LSTM, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import seaborn as sn




my_model = load_model('Video-skeleton_model.h5')    # Load model using weight file.




# Read the CSV file for prediction. 

df = pd.read_csv("file.csv")              # CSV file from open pose technique. (data to predict)

x  = df.iloc[0: , 1:]




# Creating batch with time_steps and features from data frame.

steps = time_steps = 60

data = []

for i in range(0, x.shape[0] - time_steps, steps):
    z = x[x.columns[0:x.shape[1]]].values[i: i + time_steps]
    data.append(z)




data_to_predict = np.array(data)             # Converting list data into numpy array.
print(data_to_predict.shape)




predict= my_model.predict_classes(data_to_predict)    # Prediction of activity data using saved model weight.
print(predict)                                        # Prints the one hot encoded label of class. (0-8)







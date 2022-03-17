#!/usr/bin/env python
# coding: utf-8

'''
This module helps to predict new data sets using a trained model
Author: Tadele Belay Tuli, Valay Mukesh Patel, 

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
import seaborn as sn


# Lists of methods 
def merge_rot_pos(df1,df2,label):
    """
    This function merges position and orientation of BVH data into CSV format
    
    The output is a concatinated data frame"""
                                                          # df1 is for rotation and df2 is for position
    df1 = df1.drop(columns=['Time'])                      # Drop the time coloumn from rotation and postion CSV file.
    df2 = df2.drop(columns=['Time']) 
    df_concat = pd.concat([df1, df2], axis=1)             # Mereging rotation and position CSV data.
    df_concat = df_concat.dropna()
    df_concat['category'] = label                         # Adding the associated lable (folder_name) to fetch postion and rotation CSV data. 
    return df_concat


def convert_dataset_to_csv(file_loc):

    """
    Function takes the file from dataset folder and convert it into CSV. 
    """
    
    for directory in glob.glob(file_loc):                      # Path of dataset directory.
        for file in glob.glob(directory+"*.bvh"):                  # Fetch each BVH file in dataset directory.
            f = file.split('/')  
            command_dir = f[0]+'/'+f[1]  
            command_file = f[2]    
            command = "bvh-converter -r " + command_file           # Load BVH to CSV converter.
            subprocess.call(command, shell=True, cwd=command_dir)  # Executing BVH TO CSV conveter command with shell.   
    #return command



def convert_CSV_into_df(file_loc):
    """ 
    Generate Panda dataframe from CSV data (rotation and position).
    """ 
    df = pd.DataFrame()                                       
    for directory in glob.glob(file_loc):                 # Selecting all the folders in dataset directory.
        d = [] # Empty list.
        f = directory.split('/')
        for file in glob.glob(directory+"*.csv"):             # Reading all the CSV files in dataset directory one by one.
            d.append(file)
        d = sorted(d)                                         # Ensures rotation and position are together
        while len(d)!=0:
            rot = d.pop(0)                                    # Rmove the header row from rotation and postion CSV.
            pos = d.pop(0) 
            df1 = pd.read_csv(rot, nrows=200)                 # Read the first 200 rows from rotation and position CSV. value can be 200 or 150.
            df2 = pd.read_csv(pos, nrows=200) 
            df_merge = merge_rot_pos(df1,df2,f[1])            # Call the mearge function to mearge fetch data of rotation and position CSV with class lable.
            df = df.append(df_merge,ignore_index=True)        # Append the merge data to panda dataframe one by one.
    return df




def main():
    """This is a main function"""

    # Get data in BVH
    convert_dataset_to_csv("Dataset/*/")

    # Get data into Panda dataframe
    new_df = convert_CSV_into_df("Dataset/*/")
    print(new_df)
    
    
    new_df = new_df.drop('category',axis = 1)                     # drop the class label coloumn from panda dataframe.
    print(new_df.shape)

    
    # Creating Batches with features and time steps for LSTM model. 
    steps = time_steps = 200                                   # The number of row data (length of activity) to be consider for training. value can be 200 or 150 
                                                
    data = [] 
    lables = [] 

    for i in range(0, new_df.shape[0] - time_steps, steps):
        z = new_df[new_df.columns[0:new_df.shape[1]]].values[i: i + time_steps]        # Create an array of time_steps (rows) and features (coloumn) according to value of time_steps. 
        annotation = stats.mode(df['category'][i: i + time_steps])                     # Saves a class labe assosiated with each array.
        annotation = annotation[0][0]
        data.append(z)                                                                 # Append an each array (batch) to 'data' list.
        lables.append(annotation)                                                      # Append each class label to 'lables' list. 


    lables = np.asarray(pd.get_dummies(lables), dtype = np.float32)                    # One hot encoding of lables in list.


    train_X, test_X, train_Y_one_hot, test_Y_one_hot = train_test_split(data, 
                                                            lables, test_size=0.30, 
                                                            random_state=42)          # spliting data in test and train set.

    # converting list data of test and train set in numpy array. 
    train_X = np.array(train_X)
    test_X = np.array(test_X)
    train_Y = np.array(train_Y_one_hot)
    test_Y = np.array(test_Y_one_hot)


    # LSTM model 

    epochs = 300
    batch_size = 8 # canbe tested for 4, 8 or 16

    model = Sequential()
    model.add(LSTM(200, input_shape=(train_X.shape[1], train_X.shape[2]), 
                                        return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(train_Y.shape[1], activation='softmax'))


    my_callbacks = tensorflow.callback=keras.callbacks.EarlyStopping(monitor='val_loss', 
                        min_delta=0, patience=7, verbose=2, mode='auto',
                        baseline=None, restore_best_weights=True)                   # Early stoppage when modle stops learning.

    model.compile(loss='categorical_crossentropy', 
                    optimizer='adamax', metrics=['accuracy'])                       # Model compliation with optimize, loss etc.

    model_training_history = model.fit(train_X, train_Y, epochs=epochs,
                                    batch_size=batch_size, verbose=1, 
                                    validation_split = 1.5, 
                                    callbacks = [my_callbacks])                     # Fitting the train set for training of model.

    test_loss, test_acc = model.evaluate(test_X, test_Y, batch_size=16, verbose=1)  # Evaluate trained model with test set.
    print('Test loss', test_loss)
    print('Test accuracy', test_acc)

    model.save('Skeleton_model.h5')                                                 # Saving the trained model as weight file.

    predictions = model.predict(test_X)                                             # Model makes predition for each batch of test set.
    y_1hot = pd.get_dummies(df['category'])

    y_true, y_pred = [], []

    for i in range(len(predictions)):
        y_pred.append(np.argmax(np.round(predictions[i])))                          # Append prediction of each batch of test set to empty list.
        y_true.append(np.argmax(np.round(test_Y_one_hot[i])))                       # Append true prediction of each batch of test dataset to empty list.



    # plot confusion matrix using matplotlib.
    cm = confusion_matrix(y_true,y_pred)
    df_cm = pd.DataFrame(cm, columns=y_1hot.columns, index=y_1hot.columns)
    plt.figure(figsize = (50,50))
    sn.set(font_scale=4.0) 
    sn.heatmap(df_cm, annot=True, cmap = "Blues", annot_kws={"size": 40}, fmt='g') 
    plt.show()
    plt.savefig('Skeleton_confusion_matrix.png')


if __name__ == '__main__':
    main()
#!/usr/bin/env python
# coding: utf-8

import glob
import os
import subprocess
import pandas as pd
import numpy as np
from scipy import stats
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




# Function to merge the CSV data with its associated class lable. 

def merge(df1,label):
    df1['category'] = label                             # Add the class lable to each row of CSV data.
    return df1




# Panda dataframe is generated from CSV data. 

df = pd.DataFrame()
for directory in glob.glob("Dataset/*/"):               # Selecting all the folders in dataset directory.
    d = []
    f = directory.split('/')
    for file in glob.glob(directory+"*.csv"):           # Reading all the CSV files in dataset directory one by one.
        d.append(file)
    while len(d)!=0:
        pos = d.pop(0)                                  # Rmove the header row from rotation and postion CSV.
        df1 = pd.read_csv(pos, nrows=60)                # Read the first 60 rows from rotation and position CSV. value can be 40 or 60.
        df_label = merge(df1,f[1])                      # Call the mearge function to mearge fetch data of CSV with class lable.
        df = df.append(df_label,ignore_index=True)      # Append the merge data to panda dataframe one by one.




# Replace the null value with 0 in panda dataframae. 

df = df.replace(np.nan, "0")
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]


new_df = df.drop('category',axis = 1)    # drop the class lable coloumn from panda dataframe.
print(new_df.shape)




# Creating Batches with features and time steps for LSTM model.

steps = time_steps = 60                                          # The number of row data (length of activity) to be consider for training. value can be 60 or 40                                                  
                                                        
data = []
lables = []

for i in range(0, new_df.shape[0] - time_steps, steps):
    z = new_df[new_df.columns[0:new_df.shape[1]]].values[i: i + time_steps]    # Create an array of time_steps (rows) and features (coloumn) according to value of time_steps.
    annotation = stats.mode(df['category'][i: i + time_steps])                 # Saves a class labe assosiated with each array.
    annotation = annotation[0][0]
    data.append(z)                                                             # Append an each array (batch) to 'data' list.
    lables.append(annotation)                                                  # Append each class label to 'lables' list. 




lables = np.asarray(pd.get_dummies(lables), dtype = np.float32)      # One hot encoding of lables in list.

train_X, test_X, train_Y_one_hot, test_Y_one_hot = train_test_split(data, lables, test_size=0.25, random_state=42)    # spliting data in test and train set.


# converting list data of test and train set in numpy array. 

train_X = np.array(train_X).astype('float32') 
test_X = np.array(test_X).astype('float32') 
train_Y = np.array(train_Y_one_hot).astype('float32') 
test_Y = np.array(test_Y_one_hot).astype('float32') 


# LSTM model 

epochs = 300
batch_size = 8 # can 4, 8 or 16

model = Sequential()
model.add(LSTM(200, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(train_Y.shape[1], activation='softmax'))



my_callbacks = tensorflow.callback=keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=2, mode='auto', baseline=None, restore_best_weights=True) # Eaarly stoppage when modle stops learning.

model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=['accuracy'])  # Model compliation with optimize, loss etc.

model_training_history = model.fit(train_X, train_Y, epochs=epochs, batch_size=batch_size, verbose=1, validation_split = 0.1, callbacks = [my_callbacks]) # Fitting the train set for training of model.




test_loss, test_acc = model.evaluate(test_X, test_Y, batch_size=16, verbose=1)  # Evaluate trained model with test set.

print('Test loss', test_loss)
print('Test accuracy', test_acc)




model.save('Video-skeleton_model.h5') # Saving the trained model as weight file.





predictions = model.predict(test_X)                              # Model makes predition for each batch of test set.
y_1hot = pd.get_dummies(df['category'])

y_true, y_pred = [], []

for i in range(len(predictions)):
    y_pred.append(np.argmax(np.round(predictions[i])))           # Append prediction of each batch of test set to empty list.
    y_true.append(np.argmax(np.round(test_Y_one_hot[i])))        # Append true prediction of each batch of test dataset to empty list.





# Method to plot confusion matrix using matplotlib.

cm = confusion_matrix(y_true,y_pred)
df_cm = pd.DataFrame(cm, columns=y_1hot.columns, index=y_1hot.columns)
plt.figure(figsize = (50,50))
sn.set(font_scale=4.0) 
sn.heatmap(df_cm, annot=True, cmap = "Blues", annot_kws={"size": 40}, fmt='g') 
plt.show()
plt.savefig('Video-skeleton_confusion_matrix.png')









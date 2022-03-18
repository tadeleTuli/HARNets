"""
Import CNN + LSTM models presented by 
https://github.com/SBoyNumber1/LSTM-video-classification
which was forked from https://github.com/harvitronix/five-video-classification-methods

This code is transfered with the same license (MIT)
"""

import tensorflow
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras.models import load_model
from models import ResearchModels
from data import DataSet
from extract_features import extract_features
import time
import os.path
import sys
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

def train(data_type, seq_length, model, saved_model=None,
          class_limit=None, image_shape=None,
          load_to_memory=False, batch_size=8, nb_epoch=10):
    
    # Saving the best model
    checkpointer = ModelCheckpoint(
        filepath=os.path.join('Dataset', 'checkpoints', model + '-' + data_type + \
            '.{epoch:03d}-{val_loss:.3f}.hdf5'),                                                          
        verbose=1,
        save_best_only=True)

    # Helper: TensorBoard
    tb = TensorBoard(log_dir=os.path.join('Dataset', 'logs', model))

    # Stop when model stop learning.
    early_stopper = EarlyStopping(patience=7)                                                             

    # Helper: Save results.
    timestamp = time.time()
    csv_logger = CSVLogger(os.path.join('Dataset', 'logs', model + '-' + 'training-' + \
        str(timestamp) + '.log'))

    # Get the data and process it.
    if image_shape is None:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit
        )
    else:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit,
            image_shape=image_shape
        )

    # Get samples per epoch.
    # Multiply by 0.7 to attempt to guess how much of data.data is the train set.
    steps_per_epoch = (len(data.data) * 0.7) // batch_size

    if load_to_memory:
        # Get data.
        X, y = data.get_all_sequences_in_memory('Dataset/train', data_type)
        X_test, y_test = data.get_all_sequences_in_memory('test', data_type)
    else:
        # Get generators.
        generator = data.frame_generator(batch_size, 'Dataset/train', data_type)
        val_generator = data.frame_generator(batch_size, 'Dataset/test', data_type)

    # Get the model.
    rm = ResearchModels(len(data.classes), model, seq_length, saved_model)
    
    # Fit!
    if load_to_memory:
        # Use standard fit.
        model_training_history = rm.model.fit(
            X,
            y,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1,
            callbacks=[tb, early_stopper, csv_logger, checkpointer],
            epochs=nb_epoch)
    else:
        # Use fit generator.
        rm.model.fit_generator(
            generator=generator,
            steps_per_epoch=steps_per_epoch,
            epochs=nb_epoch,
            verbose=1,
            callbacks=[tb, early_stopper, csv_logger, checkpointer],
            validation_data=val_generator,
            validation_steps=40,
            workers=4)
    
  
    #model_1 = load_model('lstm-features.027-0.616.hdf5')
    #test_loss, test_acc = rm.evaluate(X_test, y_test , batch_size=16, verbose=1)

    #print('Test loss', test_loss)
    #print('Test accuracy', test_acc)
    
    #predictions = rm.model.predict(X_test)
    #predictions = model_1.predict(X_test)
    
    #y_true, y_pred = [], []
    #for i in range(len(predictions)):
    #   y_pred.append(np.argmax(np.round(predictions[i])))
    #   y_true.append(np.argmax(np.round(y_test[i])))
    

    #cm = confusion_matrix(y_true,y_pred)
    #df_cm = pd.DataFrame(cm, columns=data.classes, index=data.classes)
    #plt.figure(figsize = (50,50))
    #sn.set(font_scale=3.0) # for label size
    #sn.heatmap(df_cm, annot=True, cmap = "Blues", annot_kws={"size": 35}, fmt='g') # font size
    #plt.show()
    #plt.savefig('cm_video_short.png')
    #plt.clf()

    #print("complete")
    
    

    """
    def plot_metric_1(metric_name_1, metric_name_2, plot_name):
      # Get Metric values using metric names as identifiers
      metric_value_1 = model_training_history.history[metric_name_1]
      metric_value_2 = model_training_history.history[metric_name_2]

      # Constructing a range object which will be used as time 
      epochs = range(len(metric_value_1))
  
      # Plotting the Graph
      plt.plot(epochs, metric_value_1, 'blue', label = metric_name_1)
      plt.plot(epochs, metric_value_2, 'red', label = metric_name_2)
  
      # Adding title to the plot
      plt.title(str(plot_name))
      plt.ylabel('loss')
      plt.xlabel('epoch')

      # Adding legend to the plot
      plt.legend()
      plt.savefig('cm_video_long_loss.png')
      plt.clf()

    def plot_metric_2(metric_name_1, metric_name_2, plot_name):
      # Get Metric values using metric names as identifiers
      metric_value_1 = model_training_history.history[metric_name_1]
      metric_value_2 = model_training_history.history[metric_name_2]

      # Constructing a range object which will be used as time 
      epochs = range(len(metric_value_1))
      plt.ylabel('accuracy')
      plt.xlabel('epoch')
  
      # Plotting the Graph
      plt.plot(epochs, metric_value_1, 'blue', label = metric_name_1)
      plt.plot(epochs, metric_value_2, 'red', label = metric_name_2)
  
      # Adding title to the plot
      plt.title(str(plot_name))

      # Adding legend to the plot
      plt.legend()
      plt.savefig('cm_video_long_acc.png')
      plt.clf()

    plot_metric_1('loss', 'val_loss', 'Total Loss vs Total Validation Loss')

    plot_metric_2('accuracy', 'val_accuracy', 'Total Accuracy vs Total Validation Accuracy')
    """

def main():
    """These are the main training settings. Set each before running
    this file."""
    
    seq_length = int(20)
    class_limit = int(2)
    image_height = int(300)
    image_width = int(500)
 
    sequences_dir = os.path.join('Dataset', 'sequences')
    if not os.path.exists(sequences_dir):
        os.mkdir(sequences_dir)

    checkpoints_dir = os.path.join('Dataset', 'checkpoints')
    if not os.path.exists(checkpoints_dir):
        os.mkdir(checkpoints_dir)

    # model can be only 'lstm'
    model = 'lstm'
    saved_model = None  # None or weights file
    load_to_memory = True # pre-load the sequences into memory
    batch_size = 16
    nb_epoch = 10
    data_type = 'features'
    image_shape = (image_height, image_width, 3)

    extract_features(seq_length=seq_length, class_limit=class_limit, image_shape=image_shape)
    train(data_type, seq_length, model, saved_model=saved_model,
          class_limit=class_limit, image_shape=image_shape,
          load_to_memory=load_to_memory, batch_size=batch_size, nb_epoch=nb_epoch)

if __name__ == '__main__':
    main()

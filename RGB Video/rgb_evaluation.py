
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
        filepath=os.path.join('data', 'checkpoints', model + '-' + data_type + \
            '.{epoch:03d}-{val_loss:.3f}.hdf5'),                                                          
        verbose=1,
        save_best_only=True)

    # Helper: TensorBoard
    tb = TensorBoard(log_dir=os.path.join('data', 'logs', model))

    # Stop when model stop learning.
    early_stopper = EarlyStopping(patience=7)                                                             

    # Helper: Save results.
    timestamp = time.time()
    csv_logger = CSVLogger(os.path.join('data', 'logs', model + '-' + 'training-' + \
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
        X, y = data.get_all_sequences_in_memory('train', data_type)
        X_test, y_test = data.get_all_sequences_in_memory('test', data_type)
    else:
        # Get generators.
        generator = data.frame_generator(batch_size, 'train', data_type)
        val_generator = data.frame_generator(batch_size, 'test', data_type)


    model_1 = load_model('lstm-features.010-0.634.hdf5')
    test_loss, test_acc = model_1.evaluate(X_test, y_test , batch_size=16, verbose=1)

    print('Test loss', test_loss)
    print('Test accuracy', test_acc)
    
    predictions = model_1.predict(X_test)
    
    y_true, y_pred = [], []
    for i in range(len(predictions)):
        y_pred.append(np.argmax(np.round(predictions[i])))
        y_true.append(np.argmax(np.round(y_test[i])))
    

    cm = confusion_matrix(y_true,y_pred)
    df_cm = pd.DataFrame(cm, columns=data.classes, index=data.classes)
    plt.figure(figsize = (25,25))
    sn.set(font_scale=4.0) # for label size
    sn.heatmap(df_cm, annot=True, cmap = "Blues", annot_kws={"size": 40}, fmt='g') # font size
    plt.savefig('cm_video_short.png')

    print("complete")



def main():
    """These are the main training settings. Set each before running
    this file."""
    
    seq_length = int(45)
    class_limit = int(2)
    image_height = int(300)
    image_width = int(500)
 
    sequences_dir = os.path.join('data', 'sequences')
    if not os.path.exists(sequences_dir):
        os.mkdir(sequences_dir)

    checkpoints_dir = os.path.join('data', 'checkpoints')
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

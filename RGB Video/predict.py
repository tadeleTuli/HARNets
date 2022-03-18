"""
Import CNN + LSTM models presented by 
https://github.com/SBoyNumber1/LSTM-video-classification
which was forked from https://github.com/harvitronix/five-video-classification-methods

This code is transfered with the same license (MIT)
"""
import os
import sys
import cv2
import numpy as np
from data import DataSet
from extractor import Extractor
from keras.models import load_model

   
seq_length = 50                                  # Input Sequance length of video frames.
class_limit = 2                                  # Number of class.
saved_model = 'lstm-features.005-0.206.hdf5'     # Trained model weight file (replace the name as per requriement)
video_file = '1.mp4'                             # Name of video file to predict activity.
 
capture = cv2.VideoCapture(os.path.join(video_file))
width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT) # float

fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter("result.avi", fourcc, 15, (int(width), int(height)))

# Get the dataset.
data = DataSet(seq_length=seq_length, class_limit=class_limit, image_shape=(height, width, 3))

# get the model.
extract_model = Extractor(image_shape=(int(height), int(width), 3))
saved_LSTM_model = load_model(saved_model)

frames = []
frame_count = 0
while True:
    ret, frame = capture.read()
    # Bail out when the video file ends
    if not ret:
        break

    # Save each frame of the video to a list
    frame_count += 1
    frames.append(frame)

    if frame_count < seq_length:
        continue # capture frames untill you get the required number for sequence
    else:
        frame_count = 0

    # For each frame extract feature and prepare it for classification
    sequence = []
    for image in frames:
        features = extract_model.extract_image(image)
        sequence.append(features)

    # Clasify sequence
    prediction = saved_LSTM_model.predict(np.expand_dims(sequence, axis=0))
    print(prediction)
    values = data.print_class_from_prediction(np.squeeze(prediction, axis=0))

    # Add prediction to frames and write them to new video
    for image in frames:
        for i in range(len(values)):
            cv2.putText(image, values[i], (40, 40 * i + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
        video_writer.write(image)

    frames = []

video_writer.release()

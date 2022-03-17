
"""
This method implements a pretrained model from caffemodel based on MPI humanpose dataset. 
MPI format has an out put of MPII Output Format: Head – 0, Neck – 1, Right Shoulder – 2, 
Right Elbow – 3, Right Wrist – 4, Left Shoulder – 5, Left Elbow – 6, Left Wrist – 7, 
Right Hip – 8, Right Knee – 9, Right Ankle – 10, Left Hip – 11, Left Knee – 12, 
Left Ankle – 13, Chest – 14, Background – 15

This code has been adapted based on: https://github.com/CMU-Perceptual-Computing-Lab/openpose

The original openpose is licensed only for academic use, and please follow their license for further use.
"""

import csv
import glob
import os
import os.path
import sys
import cv2
import time
import numpy as np
import argparse
import pandas as pd 


def load_pretrained_network():
    """
    Load the pretrained model from a folder, and return the network. 
    Caffe model comprise two files; prototxt and caffemodel file which specifies
    the neural network architecture and stores the weights of the model.

    Download the models from: https://github.com/CMU-Perceptual-Computing-Lab/openpose
    And save it under pose/mpi/ folder
    """

    protoFile = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
    weightsFile = "pose/mpi/pose_iter_160000.caffemodel"
    pre_network = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    return pre_network


def prepare_network_inputData(frame):
    """
    This function prepares network input data frame for evaluation"""

    # Input dimension parameters
    inWidth = 368
    inHeight = 368
    input_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                (0, 0, 0), swapRB=False, crop=False)
    return input_blob


def main():

    pre_network = load_pretrained_network()
    pre_network.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
    print("Using CPU device")

    # No of joints and skeleton hierarchy
    nJoints = 15
    POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]

    for directory in glob.glob("Activity/*/*mp4"):

        print("The current directory is: %s", directory)
        f = directory.split('/')
        command_file = f[1]+'_'+f[2]
        name = command_file.split('.m')[0]

        threshold = 0.1
        list1 = []
        list2 = []

        cap = cv2.VideoCapture(directory)
        hasFrame, frame = cap.read()

        # Looping over the video frames;
        
        """The loop over the video function is implemented following 
        https://github.com/CMU-Perceptual-Computing-Lab/openpose
        
        The code is not included due to license inconsistency
        """
        


if __name__ == '__main__':
    main()
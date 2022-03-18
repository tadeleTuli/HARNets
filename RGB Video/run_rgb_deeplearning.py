import glob
import os
import os.path
import sys

"""
Import CNN + LSTM models presented by 
https://github.com/SBoyNumber1/LSTM-video-classification
which was forked from https://github.com/harvitronix/five-video-classification-methods

This code is transfered with the same license (MIT)

"""
from Dataset.extract_frames import extract_files
from rgb_training import main as train
from rgb_evaluation import main as evaluate
from predict import main as predict

def main():

    # Data preparation
    dataset_folder = ['Dataset/train', 'Dataset/test']
    extract_files(dataset_folder, 'mp4') 
    
    # Uncomment and run step by step
    # Data training
    #train()
    
    # Data Evaluation
    #evaluate()
    
    # Data prediction
    #predict()
    
    
if __name__ == '__main__':
    main()
    

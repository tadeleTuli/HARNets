##__Following are the steps for training a model CNN + LSTM and prediction for video RGB data__##


1. Place the videos in data/train and data/test folders. Each video type should have its own folder as shown below.

	| data/test
		| Assembly
                    |Activity_sample_01.mp4
                    |Activity_sample_02.mp4
                               :
		| picking_front
                    |Activity_sample_01.mp4
                    |Activity_sample_02.mp4
                               :
	| data/train
		| Assembly
                    |Activity_sample_01.mp4
                    |Activity_sample_02.mp4
                               :
		| Picking_front
                    |Activity_sample_01.mp4
                    |Activity_sample_02.mp4
                               :
		...


2. Extract frames from video using script extract_files.py in data folder.

`	$ python extract_frames.py`


3. Following Parameter can be set as per the user requirement in rgb_traning.py before traning

         i) sequence_length = 50 or 20 or 70 - should be minimum to process all data. defult is 50
         ii) class_limit = 9 - No. of classes
         iii) image_height = 300 - height of video. defult is 300.
         iv) image_width args = 500 - width of video frame. defult is 500. 


4. Run train.py script for traning of CNN + lSTM model. with sequence_length, class_limit, image_height, image_width args
  
`	 $ python rgb_training.py`

5. The best model will be saved in Data/Chekpoints file.


6. To evaluate model on test dataset and get a confusion matrix run the (Note: before running python script please copy the saved model to same folder and change the name of model to copied one in script)

`	 $ python rgb_evaluation.py` 

  
 


##__Prediction__##

1. To predict the activity from video, copy the tranined weight file to main directory.

2. Set the following paprameter in predict.py file

        sequence_length = Sequences of frame to be process (should be equal to value during training).
        class_limit = No. of Class to be consider for training.
        saved_model_file = Name of trained model weight.
        video_filename = Name of video file you want to predict.

3. Use predict.py script to predict the activity from video.

`	$ python predict.py`




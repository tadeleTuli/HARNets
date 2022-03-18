
##__Following are the steps for training a model (LSTM) and prediction for video-skelton data Using Open Pose technique__##

1. First to obtain Skeleton coordinates of body joint in csv fromat from video files of activity we use open pose technique.

    1.1. Go to Open Pose Folder.

    1.2. Place the Activity data (.mp4) files in Activity folder in following fashion.
     

       | Activity/
           | Assemble_system/
                |Activity_sample_01.mp4
                |Activity_sample_02.mp4
                           :
 

    1.3. Run OpenPose.py (input=.mp4 file, output=skeleton data of each .mp4 file in csv format)

`	$ python OpenPose.py`

    
    1.4. Convert the all class of activity data, one by one using open pose.




2. Place all the converted csv files in Dataset folders shorted by class name folder as shown below.

	  | Dataset/
              | Assemble_system/
                   |Activity_sample_01.csv
                   |Activity_sample_02.csv
                             :
              | Picking_front/
                   |Activity_sample_01.csv
                   |Activity_sample_02.csv
                             :
              | Turn_Sheets/
                   |Activity_sample_01.csv
                   |Activity_sample_02.csv
                             :
		...



3. Following Parameter can be set as per the user requirement.

        i) nrows = 60 or 40 - No. of rows to read from the each CSV. defult is 60
        ii) time_steps = 60 or 40 - decides the row data to be consider for training (length of the activity.)
        iii) Batch size = 4, 8 or 16. defult is 8
        iv) epochs = As per user. defult is 300.
        Note: nrows and time_steps should be equal for good results.



4. After setting the above papramete manually Run the Video-Skeleton_LSTM_Training.py to train the LSTM model for Skeleton data.

`	$ python Video-skeleton_LSTM_Training.py`



5. The above python script will save the trained model weight as 'Video-skeleton_model.h5', plot and saves confusion matrix, traning accuracy and traning loss graph 




##__Prediction__##


1. Copy the saved model weight file (.h5) to same directory which contain Prediction.py python script. 

2. Convert mp4 (video) file to csv using openpose  (csv file should be in same directory as python script)

3. Set the name of csv file for activity prediction in python script

4. Run the prediction python script to get a prediction of classes. 

`	$ python Video-skeleton_Prediction.py`




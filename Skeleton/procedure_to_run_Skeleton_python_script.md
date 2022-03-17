
## __Following are the steps for training a model (LSTM) and prediction for Skeleton data__ ##



1. Place the Skeleton BVH files in Dataset folders. Each BVH file should have its own folder as shown below.

	| Dataset/
              | Assemble_system/
                   |Activity_sample_01.bvh
                   |Activity_sample_02.bvh
                             :
              | Picking_front/
                   |Activity_sample_01.bvh
                   |Activity_sample_02.bvh
                             :
              | Turn_Sheets/
                   |Activity_sample_01.bvh
                   |Activity_sample_02.bvh
                             :
		...



2. Following Parameter can be set as per the user requirement.

        i) nrows = 200 or 150 - No. of rows to read from the each CSV. defult is 200.
        ii) time_steps = 200 or 150 - Decides the row data to be consider for traning (length of the activity.)
        iii) Batch size = 4, 8 or 16. defult is 8
        iv) epochs = As per user. defult is 300.
        Note: nrows and time_steps should be equal for good results.



2. After setting the above papramete manually Run the 'Skeleton_LSTM_Training.py' to train the LSTM model for Skeleton data.

`	$ python Skeleton_LSTM_Training.py`



3. The above python script will save the trained model weight as 'Skeleton_model.h5', plot and saves confusion matrix, traning accuracy and traning loss graph 




## __Prediction__ ##


1. Copy the saved model weight file (.h5) to same directory which contain Skeleton_Prediction.py python script. 

2. Put the BVH file you want to get predicted into 'file_to_predict' folder.

3. Run the 'Skeleton_prediction.py' python script to get a prediction of classes. 

`	$ python Skeleton_Prediction.py`




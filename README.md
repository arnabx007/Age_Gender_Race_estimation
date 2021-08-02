# Age, Gender & Race Estimation in Real Time

### Estimate age, gender and ethnicity of faces in real time with your webcam. 
The faces are detected with opencv's **haarcascade frontalface classifier** and fed to the trained **face_net**.

The **face_net** model is trained with the architecture of **DenseNet169** with 3 branches sitting on top for age, gender and ethnicity estimation. This model has accuracies of:

>93% on gender detection
>
>87% on ethnicity estimation
>
>81% on age estimation.

The face_net .h5 model is stored in models folder.

### Here is a demo of the model on some random images :
![alt text](https://github.com/arnabx007/age_gender_race_estimation/blob/master/sample.gif "")


## Running the Model on Live Webcam Feed:
Requirements:

`pip install opencv-python`

`pip install tensorflow==2.5.0`

Run `python3 run.py`


## Training Model 
The training procedure of the model is in the `Training_Multi_Output_Network_with_Keras.ipynb` notebook and trained on GPU. The notebook is also available on Kaggle with all the outputs. 
For viewing the notebook properly, accessing dataset, easy forking and training:

https://www.kaggle.com/arnabs007/training-multi-output-model-with-keras



 

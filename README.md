# Age, Gender & Race Estimation in Real Time

### Estimate age, gender and ethnicity of faces in real time with your webcam. 

The faces are detected with opencv's `haarcascade frontalface classifier` and fed to the trained `face_net`.

The `face_net` model is trained with the architecture of **DenseNet169** with 3 branches sitting on top for age, gender and ethnicity estimation. This model has an accuracy of 93% for gender detection and 87% for ethnicity estimation and 81% for age estimation.

Both the cascade classisifer and face_net is stored in models folder.

Run `python3 run.py` for running the model on live webcam feed.

### Here is a demo of the model on some random images :


The training procedure of the model is in the `Training_Multi_Output_Network_with_Keras.ipynb` notebook and trained on GPU. The notebook is also available on Kaggle with all the outputs. 
For viewing the notebook properly, easy forking and training:






 

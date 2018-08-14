# mouth_movement_detection

A mini project detecting the mouth movement times from the video. 

This project used a classic shape ratio calculation method, in order to capture the mouth shape change when it moves.

The project is based on opencv and dlib.

To run the model
1. fetch the dlib shape_predictor model, and save in the same directory:
https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat

2. run the mouth_counter_go.py
```
python mouth_counter_go.py
```

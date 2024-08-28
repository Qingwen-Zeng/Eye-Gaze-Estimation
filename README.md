# Evaluation of Eye Gaze Estimation Methods and Their Impact on Face Recognition Performance

Gaze estimation is an important topic in biometrics, and accurate prediction of
gaze direction can improve the performance of face recognition systems. In this study,
we focus on comparing the performance of two gaze estimation methods: Convolutional
Neural Networks (CNN) and Support Vector Regression (SVR), and test the effect of eye
gaze on face recognition through the Error and Discard Characteristic (EDC) curves.
CNN models use deep learning techniques to automatically extract features from images,
providing strong predictability for gaze estimation. On the other hand, the traditional machine learning method SVR transforms the input data into a high-dimensional space to
capture complex patterns with the ability to fit nonlinear problems. By leveraging the labeled faces in the wild (LFW) dataset and applying data augmentation to break through
the limitation of sample size, we train the two models and evaluate their performance in
eye gaze estimation.
In addition, we study the effect of gaze quality on face recognition accuracy using error
versus discarded characteristic (EDC) curves. The EDC curves plotted by plotting gaze
strength as a quality score provide insights into the effectiveness of maintaining recognition accuracy under different gaze strength conditions.

The project structure is as follows:
## Data Augmentation
**Script:** `Data_Augmentation.py`

**Input:**
- `train_face`
- `val`

**Output:**
- `train_face_au`
- `val_au`

## Gaze Estimation

### CNN
**Script:** `Gaze_Estimation_CNN.py`

**Input:**
- `train_face_au`
- `val_au`
- `train_face_au.csv`
- `val_au.csv`

**Output:**: MSE and Angular Error plot, True value VS Predicted value scatter plot
- `val_result_CNN.csv`
- `best_gaze_cnn_model.pth`


### SVR
**Script:** `Gaze_Estimation_SVR.py`

**Input:**
- `train_face_au`
- `val_au`
- `train_face_au.csv`
- `val_au.csv`

**Output:**: MSE and Angular Error plot, True value VS Predicted value scatter plot
- `val_result_SVR.csv`
- `svr_model_y.pkl`
- `svr_model_x.pkl`


## Eye Gaze Impact in Face Recognition

### Create Simple Pair and Calculate the Similarity Scores
**Script:** `Similarity_Scores.py`

**Input:**
- `val_au`
- `val_au.csv`

**Output:**
- `comparison_scores.csv`

### Calculate the Quality Scores
**Script:** `Gaze_Strength_Scores.py`

**Input:**
- `shape_predictor_68_face_landmarks.dat`
- `val_au`

**Output:**
- `gaze_intensity_results.csv`

### DET
**Script:** `DET.py`

**Input:**
- `comparison_scores.csv`

**Output:**
- DET curve

### Face Recognition
**Script:** `Face_Recognition.py`

**Input:**
- `val_au`
- `comparison_scores.csv`

**Output:** Plot of Loss and Accuracy changes in Face Recognition process 
- `all_predictions.csv`


### EDC
**Script:** `EDC.py`

**Input:**
- `Final_Result.csv` (merged from `all_predictions.csv`, `comparison_scores.csv`, and `gaze_intensity_results.csv`)

**Output:**
- EDC curve

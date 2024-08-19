import cv2
import dlib
import numpy as np
import os
import pandas as pd

# load the model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
# the eye point
LEFT_EYE_POINTS = list(range(36, 42))
RIGHT_EYE_POINTS = list(range(42, 48))
#Calculate average eye aspect ratio
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear
def process_image(image_path):
    # Read the image and convert it to grayscale
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    for (i, face) in enumerate(faces):
        # locate the face feature point
        shape = predictor(gray, face)
        shape = np.array([[p.x, p.y] for p in shape.parts()])
        # Extract feature points of the eye area
        left_eye = shape[LEFT_EYE_POINTS]
        right_eye = shape[RIGHT_EYE_POINTS]
        # Calculate average eye aspect ratio
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        # Calculate average eye aspect ratio
        avg_ear = (left_ear + right_ear) / 2.0
        return avg_ear
    return None
# save the result
image_dir = 'val_au'
results = []
for image_name in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_name)
    avg_ear = process_image(image_path)
    if avg_ear is not None:
        results.append({'image': image_name, 'gaze_intensity': avg_ear})
results_df = pd.DataFrame(results)
results_df.to_csv('gaze_intensity_results.csv', index=False)

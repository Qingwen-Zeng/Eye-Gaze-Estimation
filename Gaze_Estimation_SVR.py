import os
import cv2
import numpy as np
import pandas as pd
import joblib
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from PIL import Image

# path information
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
MODEL_PATH_X = 'svr_model_x.pkl'
MODEL_PATH_Y = 'svr_model_y.pkl'
TRAIN_CSV_PATH = 'train_face_au.csv'
VAL_CSV_PATH = 'val_au.csv'
TRAIN_IMAGE_DIR = 'train_face_au'
VAL_IMAGE_DIR = 'val_au'
RESULT_CSV_PATH = 'val_result_SVR.csv'
train_df = pd.read_csv(TRAIN_CSV_PATH)
val_df = pd.read_csv(VAL_CSV_PATH)
# transform to 224*224
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# Use the pretrained resnet50 model to get the feature The last fully connected layer is removed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet50(pretrained=True).to(device)
model.eval()
model = torch.nn.Sequential(*(list(model.children())[:-1]))
# load the images according to the csv file
def load_images(df, image_dir):
    images = []
    for img_name in df['image_name']:
        img_path = os.path.join(image_dir, img_name)
        image = cv2.imread(img_path)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                feature = model(image).cpu().numpy().flatten()
            images.append(feature)
    return np.array(images)
train_images = load_images(train_df, TRAIN_IMAGE_DIR)
val_images = load_images(val_df, VAL_IMAGE_DIR)
train_gaze_x = train_df['gaze_x'].values
train_gaze_y = train_df['gaze_y'].values
val_gaze_x = val_df['gaze_x'].values
val_gaze_y = val_df['gaze_y'].values
# standardize the da
scaler = StandardScaler()
train_images = scaler.fit_transform(train_images)
val_images = scaler.transform(val_images)
# train model
svr_x = SVR(kernel='rbf')
svr_y = SVR(kernel='rbf')
svr_x.fit(train_images, train_gaze_x)
svr_y.fit(train_images, train_gaze_y)
# store model
joblib.dump(svr_x, MODEL_PATH_X)
joblib.dump(svr_y, MODEL_PATH_Y)
# load model
svr_x = joblib.load(MODEL_PATH_X)
svr_y = joblib.load(MODEL_PATH_Y)
# predict in the test set
pred_gaze_x = svr_x.predict(val_images)
pred_gaze_y = svr_y.predict(val_images)
# calculate the se and angular error
def angular_error(y_true, y_pred):
    cos_sim = np.sum(y_true * y_pred, axis=1) / (np.linalg.norm(y_true, axis=1) * np.linalg.norm(y_pred, axis=1))
    return np.arccos(np.clip(cos_sim, -1.0, 1.0))
results = []
for i in range(len(val_images)):
    true_value = np.array([val_gaze_x[i], val_gaze_y[i]])
    predicted_value = np.array([pred_gaze_x[i], pred_gaze_y[i]])
    se = np.sum((true_value - predicted_value) ** 2)
    angular_err = angular_error(true_value.reshape(1, -1), predicted_value.reshape(1, -1))[0]
    results.append([val_df.iloc[i, 0], true_value[0], true_value[1], predicted_value[0], predicted_value[1], se, angular_err])
results_df = pd.DataFrame(results, columns=['image_name', 'true_gaze_x', 'true_gaze_y', 'pred_gaze_x', 'pred_gaze_y', 'se', 'angular_error'])
results_df.to_csv(RESULT_CSV_PATH, index=False)
# calculate the mse and mean angular error
mse_x = mean_squared_error(val_gaze_x, pred_gaze_x)
mse_y = mean_squared_error(val_gaze_y, pred_gaze_y)
mean_angular_error = np.mean(results_df['angular_error'])
accuracy_x = r2_score(val_gaze_x, pred_gaze_x)
accuracy_y = r2_score(val_gaze_y, pred_gaze_y)
print(f'MSE for gaze_x: {mse_x}')
print(f'MSE for gaze_y: {mse_y}')
print(f'Mean Angular Error: {mean_angular_error}')
print(f'R^2 Score for gaze_x: {accuracy_x}')
print(f'R^2 Score for gaze_y: {accuracy_y}')
#plot the scatter
plt.figure(figsize=(10, 10))
plt.scatter(results_df['true_gaze_x'], results_df['true_gaze_y'], color='blue', label='True Values', alpha=0.5)
plt.scatter(results_df['pred_gaze_x'], results_df['pred_gaze_y'], color='red', label='Predicted Values', alpha=0.5)
plt.xlabel('Gaze X')
plt.ylabel('Gaze Y')
plt.legend()
plt.title('True vs Predicted Gaze Points')
plt.show()

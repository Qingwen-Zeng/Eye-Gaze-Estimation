import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import acos, degrees


# seed = 42
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)

#GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Path information
train_dir = 'train_face_au'
val_dir = 'val_au'
train_label_path = 'train_face_au.csv'
val_label_path = 'val_au.csv'
best_model_path = 'best_gaze_cnn_model.pth'#the path to store the best model
train_labels = pd.read_csv(train_label_path)
val_labels = pd.read_csv(val_label_path)
# read the information of images and true label
class GazeDataset(Dataset):
    def __init__(self, image_dir, labels, transform=None):
        self.image_dir = image_dir
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.labels.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        gaze = self.labels.iloc[idx, 1:3].values.astype('float32')  # 获取gaze_x和gaze_y
        if self.transform:
            image = self.transform(image)
        return image, gaze
# transform from 250*250 to 64*64
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# load train set and test set
train_dataset = GazeDataset(train_dir, train_labels, transform=transform)
val_dataset = GazeDataset(val_dir, val_labels, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)
# The CNN model
class GazeCNN(nn.Module):
    def __init__(self):
        super(GazeCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 2)  # Predict gaze_x and gaze_y
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
model = GazeCNN().to(device)
# choose loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
#calculate angular error
def angular_error(predictions, targets):
    cos_sim = np.dot(predictions, targets) / (np.linalg.norm(predictions) * np.linalg.norm(targets))
    cos_sim = np.clip(cos_sim, -1.0, 1.0)  # Clip to avoid numerical errors
    angle = degrees(acos(cos_sim))
    return angle
# train process
def train_model():
    num_epochs = 50
    best_val_loss = float('inf')
    # store the data to plot
    history = {'train_loss': [], 'val_loss': [], 'val_mse': [], 'val_angular_error': []}
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for inputs, labels in train_loader:#load train data to CNN
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss = train_loss / len(train_dataset)
        history['train_loss'].append(train_loss)
        model.eval()
        val_loss = 0
        val_mse = 0
        val_angular_err = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():#load test data to CNN
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                val_mse += mean_squared_error(labels.cpu().numpy(), outputs.cpu().numpy()) * inputs.size(0)
                for i in range(len(labels)):
                    angular_err = angular_error(outputs[i].cpu().numpy(), labels[i].cpu().numpy())
                    val_angular_err += angular_err
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        val_loss = val_loss / len(val_dataset)
        val_mse = val_mse / len(val_dataset)
        val_angular_err = val_angular_err / len(val_dataset)
        history['val_loss'].append(val_loss)
        history['val_mse'].append(val_mse)
        history['val_angular_error'].append(val_angular_err)
        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'Val MSE: {val_mse:.4f}, Val Angular Error: {val_angular_err:.4f}')
        # store the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print("Best model saved to disk.")
    # plot the MSE and Anguar error changes
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='train loss')
    plt.plot(history['val_loss'], label='val loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(history['val_mse'], label='val mse')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.subplot(1, 3, 3)
    plt.plot(history['val_angular_error'], label='val angular error')
    plt.xlabel('Epochs')
    plt.ylabel('Angular Error (degrees)')
    plt.legend()
    plt.show()

def evaluate_model():
    # load the best model
    model.load_state_dict(torch.load(best_model_path))
    # predict the test set and the error
    model.eval()
    val_preds = []
    val_labels = []
    val_mse = 0
    val_angular_err = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            val_mse += mean_squared_error(labels.cpu().numpy(), outputs.cpu().numpy()) * inputs.size(0)
            for i in range(len(labels)):
                angular_err = angular_error(outputs[i].cpu().numpy(), labels[i].cpu().numpy())
                val_angular_err += angular_err
            val_preds.extend(outputs.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
    val_mse = val_mse / len(val_dataset)
    val_angular_err = val_angular_err / len(val_dataset)
    df = pd.DataFrame({
        'Image Name': val_labels['image_name'],
        'True Gaze X': [label[0] for label in val_labels],
        'True Gaze Y': [label[1] for label in val_labels],
        'Predicted Gaze X': [pred[0] for pred in val_preds],
        'Predicted Gaze Y': [pred[1] for pred in val_preds],
        'MSE': [val_mse] * len(val_labels),
        'Angular Error': [val_angular_err] * len(val_labels)
    })
    df.to_csv('val_result_CNN.csv', index=False) #data store in val_result_CNN.csv


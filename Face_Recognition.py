import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Define the Dataset
class FaceDataset(Dataset):
    def __init__(self, csv_file, image_folder, transform=None):
        self.pairs = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_name = self.pairs.iloc[idx, 0]
        img2_name = self.pairs.iloc[idx, 1]
        img1_path = os.path.join(self.image_folder, img1_name)
        img2_path = os.path.join(self.image_folder, img2_name)
        label = 1 if self.pairs.iloc[idx, 2] == self.pairs.iloc[idx, 3] else 0

        image1 = Image.open(img1_path).convert('RGB')
        image2 = Image.open(img2_path).convert('RGB')

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return image1, image2, torch.tensor(label, dtype=torch.float32), img1_name, img2_name
# Define transformations without data augmentation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# Create dataset
csv_path = 'comparison_scores.csv'
image_folder = 'val_au'  # Modify this path as needed
dataset = FaceDataset(csv_file=csv_path, image_folder=image_folder, transform=transform)
# Split the dataset into training and testing sets
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# Define the simplified CNN Model
class SimpleSiameseNetwork(nn.Module):
    def __init__(self):
        super(SimpleSiameseNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2ford(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Linear(256 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 1)
    def forward_one(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        dist = torch.abs(out1 - out2)
        out = self.fc2(dist)
        return torch.sigmoid(out)
model = SimpleSiameseNetwork().cuda()  # Move model to GPU if available
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
# Function to evaluate the model
def evaluate(model, dataloader):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images1, images2, labels, _, _ in dataloader:
            images1, images2, labels = images1.cuda(), images2.cuda(), labels.cuda()
            outputs = model(images1, images2)
            loss = criterion(outputs.squeeze(), labels)
            running_loss += loss.item()
            predicted = (outputs.squeeze() > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# Train the Model
num_epochs = 10
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
        for images1, images2, labels, _, _ in train_dataloader:
            images1, images2, labels = images1.cuda(), images2.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(images1, images2)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            predicted = (outputs.squeeze() > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pbar.update(1)
    epoch_loss = running_loss / len(train_dataloader)
    epoch_acc = correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)
    test_loss, test_acc = evaluate(model, test_dataloader)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
# Step 4: Plot Loss and Accuracy
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(test_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.savefig('training_and_validation_metrics.png')
plt.show()
# Step 5: Make Predictions on Train and Test Sets and Save to CSV
model.eval()
all_predictions = []
# Helper function for making predictions
def predict(dataloader):
    predictions = []
    with torch.no_grad():
        for images1, images2, labels, img1_names, img2_names in dataloader:
            images1, images2, labels = images1.cuda(), images2.cuda(), labels.cuda()
            outputs = model(images1, images2)
            predicted = (outputs.squeeze() > 0.5).float()
            predictions.extend(zip(img1_names, img2_names, labels.cpu().numpy(), predicted.cpu().numpy()))
    return predictions
# Make predictions on the training set
train_predictions = predict(train_dataloader)
all_predictions.extend(train_predictions)
# Make predictions on the test set
test_predictions = predict(test_dataloader)
all_predictions.extend(test_predictions)
# Save all predictions to CSV
output_df = pd.DataFrame(all_predictions, columns=['Image1', 'Image2', 'TrueLabel', 'PredictedLabel'])
output_df.to_csv('all_predictions.csv', index=False)



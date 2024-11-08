# %%
import os
import random
import shutil
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms
import torchvision.transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder # that can be applied on these datasets
import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm
import os
import random
import shutil
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import matplotlib.pyplot as plt

# https://www.kaggle.com/code/vikasbhadoria/mnist-data-99-5-accuracy-using-pytorch/

# data/classifier/
# ├── train/
# │   ├── class_0/  (contains images for class 0)
# │   ├── class_1/  (contains images for class 1)
# │   ├── .../
# └── val/
#     ├── class_0/
#     ├── class_1/
#     ├── .../




# # Function to create directories for training and validation
# def create_dirs(train_dir, val_dir):
#     os.makedirs(train_dir, exist_ok=True)
#     os.makedirs(val_dir, exist_ok=True)
#     for i in range(10):
#         os.makedirs(os.path.join(train_dir, f'class_{i}'), exist_ok=True)
#         os.makedirs(os.path.join(val_dir, f'class_{i}'), exist_ok=True)

# # Function to classify files by their labels (from filename)
# def classify_files(input_dir):
#     png_files = list(Path(input_dir).rglob('*.png'))
#     class_files = {str(i): [] for i in range(10)}
#     for file in png_files:
#         class_label = file.stem.split('_')[0]
#         if class_label in class_files:
#             class_files[class_label].append(file)
#     return class_files

# # Function to move files to their corresponding directories
# def move_files(class_files, train_dir, val_dir, total_train_size=1000, total_val_size=5000, deviation_limit=30):
#     success_count = 0
#     failure_count = 0
#     failed_files = []

#     # Define train/val size per class
#     train_size_per_class = total_train_size // 10
#     val_size_per_class = total_val_size // 10
    
#     # Process each class and move files
#     for class_label, files in tqdm(class_files.items(), desc="Processing classes", total=10):
#         random.shuffle(files)

#         class_total = len(files)
#         val_count = min(val_size_per_class, class_total - train_size_per_class)
#         train_count = class_total - val_count

#         # Adjust file distribution to meet deviation limit
#         if abs(train_count - train_size_per_class) > deviation_limit:
#             adjustment = (train_count - train_size_per_class) // abs(train_count - train_size_per_class)
#             train_count = train_size_per_class + adjustment * deviation_limit
#             val_count = class_total - train_count

#         val_files = files[:val_count]
#         train_files = files[val_count:val_count + train_count]

#         # Move validation files
#         for file_path in tqdm(val_files, desc=f"Moving validation files for class {class_label}", leave=False):
#             dst_path = os.path.join(val_dir, f'class_{class_label}', file_path.name)
#             try:
#                 shutil.copy(file_path, dst_path)
#                 success_count += 1
#             except Exception as e:
#                 failure_count += 1
#                 failed_files.append((file_path, str(e)))  # Store failed file and error

#         # Move training files
#         for file_path in tqdm(train_files, desc=f"Moving training files for class {class_label}", leave=False):
#             dst_path = os.path.join(train_dir, f'class_{class_label}', file_path.name)
#             try:
#                 shutil.copy(file_path, dst_path)
#                 success_count += 1
#             except Exception as e:
#                 failure_count += 1
#                 failed_files.append((file_path, str(e)))  # Store failed file and error

#     return success_count, failure_count, failed_files

# Function to create DataLoader objects
def create_dataloaders(train_dir, val_dir, batch_size=100):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Create datasets
    training_dataset = ImageFolder(root=train_dir, transform=transform)
    validation_dataset = ImageFolder(root=val_dir, transform=transform)
    
    # Subset of validation set (if needed)
    val_dataset_size = len(validation_dataset)
    subset_indices = np.random.choice(val_dataset_size, size=5000, replace=False)  # Subsample 5,000 images
    validation_subset_dataset = Subset(validation_dataset, subset_indices)

    # Create DataLoaders
    training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_subset_dataset, batch_size=batch_size, shuffle=True)
    
    return training_loader, validation_loader

# LeNet model class
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(50 * 5 * 5, 500)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

# Function to train the model
def train_model(model, criterion, optimizer, training_loader, validation_loader, epochs=15, device='cuda'):
    running_loss_history = []
    running_corrects_history = []
    val_running_loss_history = []
    val_running_corrects_history = []

    for e in range(epochs):
        running_loss = 0.0
        running_corrects = 0.0
        val_running_loss = 0.0
        val_running_corrects = 0.0

        for inputs, labels in training_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data)

        # Validation phase
        with torch.no_grad():
            for val_inputs, val_labels in validation_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_labels)
                _, val_preds = torch.max(val_outputs, 1)
                val_running_loss += val_loss.item()
                val_running_corrects += torch.sum(val_preds == val_labels.data)

        epoch_loss = running_loss / len(training_loader)
        epoch_acc = running_corrects.float() / len(training_loader)
        val_epoch_loss = val_running_loss / len(validation_loader)
        val_epoch_acc = val_running_corrects.float() / len(validation_loader)

        running_loss_history.append(epoch_loss)
        running_corrects_history.append(epoch_acc)
        val_running_loss_history.append(val_epoch_loss)
        val_running_corrects_history.append(val_epoch_acc)

        print(f"Epoch {e+1}/{epochs}")
        print(f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}")
        print(f"Val Loss: {val_epoch_loss:.4f}, Val Accuracy: {val_epoch_acc:.4f}")

    return running_loss_history, running_corrects_history, val_running_loss_history, val_running_corrects_history


def plot_results(running_loss_history, val_running_loss_history, running_corrects_history, val_running_corrects_history):
    # Fixed axis limits for consistency across plots
    loss_ylim = (0, 1)  # Limit for loss (0 to max loss)
    acc_ylim = (0, 100)  # Limit for accuracy (0 to 1)

    plt.figure(figsize=(12, 6))
    
    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(running_loss_history, label='Training Loss')
    plt.plot(val_running_loss_history, label='Validation Loss')
    plt.legend()
    plt.title('Loss over Epochs')
    plt.ylim(loss_ylim)  # Apply fixed y-axis range for loss
    
    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(running_corrects_history, label='Training Accuracy')
    plt.plot(val_running_corrects_history, label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy over Epochs')
    plt.ylim(acc_ylim)  # Apply fixed y-axis range for accuracy

    plt.tight_layout()  # Adjust subplots to avoid overlap
    plt.show()


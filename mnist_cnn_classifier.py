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

# https://www.kaggle.com/code/vikasbhadoria/mnist-data-99-5-accuracy-using-pytorch/


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # To use to cuda GPU
# print(device)


import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy as np

def create_dataloaders(train_dir, val_dir, batch_size=100, val_subset_size=5000, image_size=(32, 32)):
    """
    Create DataLoaders for training and validation datasets with optional validation subset.

    Parameters:
    - train_dir (str): Path to the training dataset folder.
    - val_dir (str): Path to the validation dataset folder.
    - batch_size (int): Batch size for DataLoader (default is 100).
    - val_subset_size (int): Number of samples to subset from the validation set (default is 5000).
    - image_size (tuple): Size to which the images will be resized (default is (32, 32)).

    Returns:
    - training_loader (DataLoader): DataLoader for training data.
    - validation_loader (DataLoader): DataLoader for validation data subset.
    """

    # Define transformations (resize, grayscale, tensor conversion, normalization)
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize the images
    ])

    # Load training and validation datasets
    training_dataset = ImageFolder(root=train_dir, transform=transform)
    validation_dataset = ImageFolder(root=val_dir, transform=transform)

    # Subset the validation dataset if required
    val_dataset_size = len(validation_dataset)
    if val_subset_size:
        subset_indices = np.random.choice(val_dataset_size, size=val_subset_size, replace=False)
        validation_subset_dataset = Subset(validation_dataset, subset_indices)
    else:
        validation_subset_dataset = validation_dataset

    # Create DataLoaders
    training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_subset_dataset, batch_size=batch_size, shuffle=True)

    return training_loader, validation_loader


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)  # Conv layer1
        self.conv2 = nn.Conv2d(20, 50, 5, 1)  # Conv layer2
        self.fc1 = nn.Linear(50 * 5 * 5, 500)  # Fully connected layer1
        self.dropout1 = nn.Dropout(0.5)  # Dropout layer
        self.fc2 = nn.Linear(500, 10)  # Fully connected layer2
    
    def forward(self, x):
        x = F.relu(self.conv1(x))  # Apply ReLU to Conv1
        x = F.max_pool2d(x, 2, 2)  # Max pooling after Conv1
        x = F.relu(self.conv2(x))  # Apply ReLU to Conv2
        x = F.max_pool2d(x, 2, 2)  # Max pooling after Conv2
        # print(x.shape)  # Debugging print statement
        batch_size = x.size(0)  # Get batch size
        x = x.view(batch_size, -1)  # Flatten the tensor for the fully connected layer

        x = F.relu(self.fc1(x))  # Fully connected layer 1 with ReLU
        x = self.dropout1(x)  # Apply dropout
        x = self.fc2(x)  # Fully connected layer 2 (output)
        return x


import torch

def train_and_evaluate(model, criterion, optimizer, training_loader, validation_loader, epochs=15, device='cpu'):
    """
    Train and evaluate the model for a specified number of epochs.

    Parameters:
    - model (torch.nn.Module): The neural network model to train.
    - criterion (torch.nn.Module): The loss function.
    - optimizer (torch.optim.Optimizer): The optimizer used to update model weights.
    - training_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
    - validation_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
    - epochs (int): Number of epochs to train the model (default is 15).
    - device (str or torch.device): The device to run the model on ('cpu' or 'cuda').

    Returns:
    - running_loss_history (list): List of average training loss per epoch.
    - running_corrects_history (list): List of training accuracy per epoch.
    - val_running_loss_history (list): List of average validation loss per epoch.
    - val_running_corrects_history (list): List of validation accuracy per epoch.
    """

    # Initialize history containers
    running_loss_history = []
    running_corrects_history = []
    val_running_loss_history = []
    val_running_corrects_history = []

    for epoch in range(epochs):
        # Initialize counters for each epoch
        running_loss = 0.0
        running_corrects = 0.0
        val_running_loss = 0.0
        val_running_corrects = 0.0

        # Set the model to training mode
        model.train()

        for inputs, labels in training_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization step
            loss.backward()
            optimizer.step()

            # Calculate the predictions
            _, preds = torch.max(outputs, 1)

            # Update running loss and corrects for training
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data)

        # Set the model to evaluation mode for validation
        model.eval()

        with torch.no_grad():  # No gradients needed for validation
            for val_inputs, val_labels in validation_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)

                # Forward pass for validation
                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_labels)

                # Calculate validation predictions
                _, val_preds = torch.max(val_outputs, 1)

                # Update running loss and corrects for validation
                val_running_loss += val_loss.item()
                val_running_corrects += torch.sum(val_preds == val_labels.data)

        # Calculate average losses and accuracies
        epoch_loss = running_loss / len(training_loader)
        epoch_acc = running_corrects.float() / len(training_loader)

        val_epoch_loss = val_running_loss / len(validation_loader)
        val_epoch_acc = val_running_corrects.float() / len(validation_loader)

        # Save to history
        running_loss_history.append(epoch_loss)
        running_corrects_history.append(epoch_acc)
        val_running_loss_history.append(val_epoch_loss)
        val_running_corrects_history.append(val_epoch_acc)

        # Print epoch statistics
        print(f'Epoch {epoch + 1}/{epochs}')
        print(f'Training loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')
        print(f'Validation loss: {val_epoch_loss:.4f}, Acc: {val_epoch_acc:.4f}')

    return running_loss_history, running_corrects_history, val_running_loss_history, val_running_corrects_history



import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

def plot_training_vs_validation(running_loss_history, val_running_loss_history, 
                                running_acc_history, val_running_acc_history, 
                                model_name, 
                                plot_title='Training vs Validation', 
                                xlabel='Epoch', ylabel='Value', 
                                plot_size=(12, 6), save_path=None):
    """
    Plot and save a comparison of training vs validation accuracy and loss over epochs, side by side.
    Includes the model name in the plot title and filename.

    Parameters:
    - running_loss_history (list): List of training loss values for each epoch.
    - val_running_loss_history (list): List of validation loss values for each epoch.
    - running_acc_history (list): List of training accuracy values for each epoch.
    - val_running_acc_history (list): List of validation accuracy values for each epoch.
    - model_name (str): The name of the model to include in the plot title and filename.
    - plot_title (str): Title of the plot (default is 'Training vs Validation').
    - xlabel (str): Label for the x-axis (default is 'Epoch').
    - ylabel (str): Label for the y-axis (default is 'Value').
    - plot_size (tuple): Size of the plot (default is (12, 6), width and height of the entire figure).
    - save_path (str): Path where to save the plot image (default is None). If None, saves using the model name.

    Returns:
    - None (saves the plot to the specified path and displays it).
    """
    # Create a figure and two subplots (axes)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=plot_size)

    # Plot training and validation accuracy on the left plot (ax1)
    ax1.plot(running_acc_history, label='Training Accuracy', color='b', linestyle='-', linewidth=2)
    ax1.plot(val_running_acc_history, label='Validation Accuracy', color='r', linestyle='--', linewidth=2)
    ax1.set_title(f'{model_name} - Accuracy', fontsize=16, fontweight='bold')
    ax1.set_xlabel(xlabel, fontsize=14)
    ax1.set_ylabel('Accuracy', fontsize=14)
    ax1.grid(True, linestyle=':', linewidth=0.5)
    ax1.legend(loc='upper left', fontsize=12, borderaxespad=0.1)

    # Plot training and validation loss on the right plot (ax2)
    ax2.plot(running_loss_history, label='Training Loss', color='b', linestyle='-', linewidth=2)
    ax2.plot(val_running_loss_history, label='Validation Loss', color='r', linestyle='--', linewidth=2)
    ax2.set_title(f'{model_name} - Loss', fontsize=16, fontweight='bold')
    ax2.set_xlabel(xlabel, fontsize=14)
    ax2.set_ylabel('Loss', fontsize=14)
    ax2.grid(True, linestyle=':', linewidth=0.5)
    ax2.legend(loc='upper right', fontsize=12, borderaxespad=0.1)

    # Adjust the layout for better aesthetics
    plt.tight_layout()

    # Set default save path if not provided
    if save_path is None:
        save_path = f'{model_name}_training_vs_validation.png'

    # Save the plot with high DPI for publications
    plt.savefig(save_path, dpi=300)

    # Display the plot
    plt.show()


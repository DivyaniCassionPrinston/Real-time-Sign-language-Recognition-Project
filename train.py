# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np

from model import ASLNet

class ASLDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        # Filter out hidden files like .DS_Store
        self.classes = [f for f in sorted(os.listdir(data_dir)) 
                       if not f.startswith('.') and os.path.isdir(os.path.join(data_dir, f))]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            if os.path.isdir(class_dir):  # Verify it's a directory
                # Filter out hidden files in the class directory
                image_files = [f for f in os.listdir(class_dir) 
                             if not f.startswith('.') and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                for img_name in image_files:
                    img_path = os.path.join(class_dir, img_name)
                    if os.path.isfile(img_path):  # Verify it's a file
                        self.images.append(img_path)
                        self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            label = self.labels[idx]
            
            if self.transform:
                image = self.transform(image)
                
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # Return a default image or raise the exception
            raise

def train_model(data_dir, num_epochs=10):
    # Verify data directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Print initial dataset information
    print(f"Loading dataset from: {data_dir}")
    
    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        # Create dataset and dataloader
        dataset = ASLDataset(data_dir, transform=transform)
        
        if len(dataset) == 0:
            raise ValueError("No valid images found in the dataset")
            
        print(f"Found {len(dataset)} images across {len(dataset.classes)} classes")
        print("Classes:", dataset.classes)
        
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Initialize model, loss function, and optimizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        model = ASLNet().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        best_val_acc = 0.0
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
            # Validation
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            val_acc = 100 * correct / total
            print(f'Epoch {epoch+1}/{num_epochs}')
            print(f'Loss: {running_loss/len(train_loader):.4f}')
            print(f'Validation Accuracy: {val_acc:.2f}%')
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 'best_model.pth')
                
        return model, dataset.classes
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise


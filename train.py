import os
import random
import shutil
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.utils.data as data
import torch
import numpy as np
from unet import UNet
from utils.metrics import generalized_dice
# Instead of torchvision.transforms.Compose, use your custom MyCompose
from utils.dataloading import get_patient_data, ACDCDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_data_dict = get_patient_data('data/preprocessed/train')
val_data_dict = get_patient_data('data/preprocessed/val')


# Rely on the dataset to pick a random slice and use standard DataLoader
train_dataset = ACDCDataset(train_data_dict)
val_dataset = ACDCDataset(val_data_dict)

model = UNet(
    in_channels=1,
    n_classes=4,
    depth=5,
    wf=48,
    padding=True,
    batch_norm=True,
    up_mode='upsample'
)

# Training configuration
MAX_ITERATIONS = 16000
BATCH_SIZE = 10
LEARNING_RATE = 1e-3
VALIDATION_STEPS = 10
# Update DataLoader batch size
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training utilities
def evaluate_model(model, val_loader):
    model.eval()
    total_ce_loss = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            
            # Calculate Cross Entropy Loss
            ce_loss = criterion(outputs, masks)
            total_ce_loss += ce_loss.item()
            
            # Convert predictions for dice calculation
            pred = torch.softmax(outputs, dim=1)
            pred = pred.cpu().numpy()
            masks = masks.cpu().numpy()
            
            # Store batch predictions and targets
            predictions.append(pred)
            targets.append(masks)
    
    # Calculate mean CE loss
    mean_ce_loss = total_ce_loss / len(val_loader)
    
    # Calculate generalized dice score
    all_preds = np.concatenate(predictions, axis=0)
    all_targets = np.concatenate(targets, axis=0)
    gdice = generalized_dice(all_preds, all_targets)
    
    return mean_ce_loss, gdice

# Training loop
best_gdice = 0
iteration = 0
train_iterator = iter(train_loader)

while iteration < MAX_ITERATIONS:
    model.train()
    
    try:
        images, masks = next(train_iterator)
    except StopIteration:
        train_iterator = iter(train_loader)
        images, masks = next(train_iterator)
    
    images, masks = images.to(device), masks.to(device)
    
    # Forward pass
    outputs = model(images)
    loss = criterion(outputs, masks)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Evaluate every 100 iterations
    if iteration % VALIDATION_STEPS == 0:
        val_ce_loss, val_gdice = evaluate_model(model, val_loader)
        print(f'Iteration {iteration}:')
        print(f'Validation CE Loss: {val_ce_loss:.4f}')
        print(f'Validation GDice: {val_gdice:.4f}')
        
        # Save best model
        if val_gdice > best_gdice:
            best_gdice = val_gdice
            torch.save({
                'iteration': iteration,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_gdice': best_gdice,
            }, 'best_model.pth')
            print(f'New best model saved! GDice: {best_gdice:.4f}')
    
    iteration += 1

print('Training completed!')





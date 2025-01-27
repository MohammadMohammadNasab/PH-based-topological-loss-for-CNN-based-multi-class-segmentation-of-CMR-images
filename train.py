import os
import random
import shutil
import datetime
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.utils.data as data
import torch
import numpy as np
from unet import UNet
from utils.metrics import generalized_dice
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

# Add tracking lists for metrics
train_losses = []
val_losses = []
val_gdice_scores = []
iterations = []

# Create directories for saving results
def create_directories():
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)

# Update plot_metrics function
def plot_metrics(train_losses, val_losses, val_gdice_scores, iterations):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Plot training loss
    plt.figure(figsize=(8, 6))
    plt.plot(iterations, train_losses, label='Train Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('plots', f'train_loss_{timestamp}.png'))
    plt.close()
    
    # Plot validation loss
    plt.figure(figsize=(8, 6))
    plt.plot(iterations, val_losses, label='Validation Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('plots', f'val_loss_{timestamp}.png'))
    plt.close()
    
    # Plot GDice scores
    plt.figure(figsize=(8, 6))
    plt.plot(iterations, val_gdice_scores, label='Validation GDice')
    plt.xlabel('Iteration')
    plt.ylabel('GDice Score')
    plt.title('Validation GDice Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('plots', f'gdice_{timestamp}.png'))
    plt.close()

# Create directories before training starts
create_directories()

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
    
    # Store training loss
    train_losses.append(loss.item())
    
    # Evaluate every VALIDATION_STEPS iterations
    if iteration % VALIDATION_STEPS == 0:
        val_ce_loss, val_gdice = evaluate_model(model, val_loader)
        print(f'Iteration {iteration}:')
        print(f'Training CE Loss: {loss.item():.4f}')
        print(f'Validation CE Loss: {val_ce_loss:.4f}')
        print(f'Validation GDice: {val_gdice:.4f}')
        
        # Store metrics
        iterations.append(iteration)
        val_losses.append(val_ce_loss)
        val_gdice_scores.append(val_gdice)
        
        # Plot current metrics
        plot_metrics(
            train_losses[::VALIDATION_STEPS],  # Sample training loss at validation steps
            val_losses,
            val_gdice_scores,
            iterations
        )
        
        # Save best model with timestamp
        if val_gdice > best_gdice:
            best_gdice = val_gdice
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            model_path = os.path.join('models', f'best_model_{timestamp}_gdice_{val_gdice:.4f}.pth')
            torch.save({
                'iteration': iteration,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_gdice': best_gdice,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_gdice_scores': val_gdice_scores,
                'iterations': iterations
            }, model_path)
            print(f'New best model saved at {model_path}! GDice: {best_gdice:.4f}')
    
    iteration += 1

print('Training completed!')
# Final plot
plot_metrics(
    train_losses[::VALIDATION_STEPS],
    val_losses,
    val_gdice_scores,
    iterations
)





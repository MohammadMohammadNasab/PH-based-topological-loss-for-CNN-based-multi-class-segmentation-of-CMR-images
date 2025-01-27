import argparse
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

def parse_args():
    parser = argparse.ArgumentParser(description='Train UNet model for ACDC segmentation')
    parser.add_argument('--train_dir', type=str, default='data/preprocessed/train',
                        help='Directory containing training data')
    parser.add_argument('--val_dir', type=str, default='data/preprocessed/val',
                        help='Directory containing validation data')
    parser.add_argument('--max_iterations', type=int, default=16000,
                        help='Maximum number of training iterations')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size for training and validation')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate for optimizer')
    parser.add_argument('--validation_steps', type=int, default=10,
                        help='Number of iterations between validations')
    return parser.parse_args()

# Move the main training logic into a function
def create_experiment_dir():
    # Create base experiments directory
    experiments_dir = 'Experiments'
    os.makedirs(experiments_dir, exist_ok=True)
    
    # Count existing experiment folders to create new one
    existing_experiments = [d for d in os.listdir(experiments_dir) if d.startswith('experiment_')]
    next_exp_num = len(existing_experiments) + 1
    
    # Create new experiment directory with subdirectories
    exp_dir = os.path.join(experiments_dir, f'experiment_{next_exp_num}')
    os.makedirs(exp_dir)
    os.makedirs(os.path.join(exp_dir, 'plots'))
    os.makedirs(os.path.join(exp_dir, 'models'))
    os.makedirs(os.path.join(exp_dir, 'logs'))
    
    return exp_dir

def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create experiment directory at the start
    exp_dir = create_experiment_dir()
    
    # Load data using argument paths
    train_data_dict = get_patient_data(args.train_dir)
    val_data_dict = get_patient_data(args.val_dir)

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

    # Use argument values for configuration
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Rest of the existing tracking lists and helper functions
    train_losses = []
    val_losses = []
    val_gdice_scores = []
    iterations = []

    # Create directories for saving results
    def create_directories():
        os.makedirs('models', exist_ok=True)
        os.makedirs('plots', exist_ok=True)

    # Update plot_metrics function
    def plot_metrics(train_losses, val_losses, val_gdice_scores, iterations, exp_dir):
        # Plot training loss
        plt.figure(figsize=(8, 6))
        plt.plot(iterations, train_losses, label='Train Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, 'plots', 'train_loss.png'))
        plt.close()
        
        # Plot validation loss
        plt.figure(figsize=(8, 6))
        plt.plot(iterations, val_losses, label='Validation Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Validation Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, 'plots', 'val_loss.png'))
        plt.close()
        
        # Plot GDice scores
        plt.figure(figsize=(8, 6))
        plt.plot(iterations, val_gdice_scores, label='Validation GDice')
        plt.xlabel('Iteration')
        plt.ylabel('GDice Score')
        plt.title('Validation GDice Score')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, 'plots', 'gdice.png'))
        plt.close()

    # Add new logging function
    def log_metrics(exp_dir, iteration, train_loss, val_loss, val_gdice, is_best=False):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file = os.path.join(exp_dir, 'logs', 'training_log.txt')
        
        log_entry = f"[{timestamp}] Iteration {iteration}\n"
        log_entry += f"Training Loss: {train_loss:.4f}\n"
        log_entry += f"Validation Loss: {val_loss:.4f}\n"
        log_entry += f"Validation GDice: {val_gdice:.4f}\n"
        
        if is_best:
            log_entry += "*** New Best Model! ***\n"
        
        log_entry += "-" * 50 + "\n"
        
        with open(log_file, 'a') as f:
            f.write(log_entry)

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

    # Training loop with argument values
    best_gdice = 0
    iteration = 0
    train_iterator = iter(train_loader)

    while iteration < args.max_iterations:
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
        
        # Evaluate using argument value for validation steps
        if iteration % args.validation_steps == 0:
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
                train_losses[::args.validation_steps],  # Sample training loss at validation steps
                val_losses,
                val_gdice_scores,
                iterations,
                exp_dir
            )
            
            # Modified model saving section
            if val_gdice > best_gdice:
                best_gdice = val_gdice
                model_path = os.path.join(exp_dir, 'models', 'best_model.pth')
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
                print(f'New best model saved! GDice: {best_gdice:.4f}')
                # Log the best model metrics
                log_metrics(exp_dir, iteration, loss.item(), val_ce_loss, val_gdice, is_best=True)
            else:
                # Log regular metrics
                log_metrics(exp_dir, iteration, loss.item(), val_ce_loss, val_gdice, is_best=False)

        iteration += 1

    print('Training completed!')
    # Final plot
    plot_metrics(
        train_losses[::args.validation_steps],
        val_losses,
        val_gdice_scores,
        iterations,
        exp_dir
    )

if __name__ == '__main__':
    main()





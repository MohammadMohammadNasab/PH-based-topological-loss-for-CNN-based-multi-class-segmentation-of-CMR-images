import argparse
import os
import random
import shutil
import datetime
import uuid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import numpy as  np
from tqdm import tqdm
from unet import UNet
from utils.dataloading import get_patient_data, TrainACDCDataset, ValACDCDataset
from metrics import generalized_dice, hausdorff_distance, dice_coefficient, betti_error, topological_success, compute_class_combinations_betti
from topo import compute_topological_loss  # Assumes you have this function implemented
from torch.cuda.amp import autocast, GradScaler  # Add this import for mixed precision
# Optionally, import a helper to load or define the anatomical prior:

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
    parser.add_argument('--use_topo_loss', action='store_true',
                        help='Flag to use topological loss during training')
    parser.add_argument('--lambda_ce', type=float, default=1.0,
                        help='Weight for the Cross Entropy loss')
    parser.add_argument('--multi_class', action='store_true',
                        help='Flag to enable multi-class segmentation')
    parser.add_argument('--attention', action='store_true',
                        help='Use attention mechanism in UNet')
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
    exp_dir = os.path.join(experiments_dir, f'experiment_{uuid.uuid4()}')
    os.makedirs(exp_dir)
    os.makedirs(os.path.join(exp_dir, 'plots'))
    os.makedirs(os.path.join(exp_dir, 'models'))
    os.makedirs(os.path.join(exp_dir, 'logs'))
    
    return exp_dir

# Define the function to get anatomical priors
def get_priors(multi_class):
    if multi_class:
        return {
            (1,):   (1, 0),
            (2,):   (1, 1),
            (3,):   (1, 0),
            (1, 2): (1, 1),
            (1, 3): (2, 0),
            (2, 3): (1, 0)
        }
    else:
        return {
            (1,):   (1, 0),
            (2,):   (1, 1),
            (3,):   (1, 0),
        }

def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create experiment directory at the start
    exp_dir = create_experiment_dir()
    
    # Load data using argument paths
    train_data_dict , _,_ = get_patient_data(args.train_dir)
    val_data_dict, val_image_paths, val_lbl_paths = get_patient_data(args.val_dir)

    train_dataset = TrainACDCDataset(train_data_dict)
    val_dataset = ValACDCDataset(val_image_paths, val_lbl_paths)

    model = UNet(
        in_channels=1,
        n_classes=4,
        depth=5,
        wf=48,
        padding=True,
        batch_norm=True,
        up_mode='upsample',
        attention=args.attention  # Pass the attention flag
    )

    # Use argument values for configuration
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scaler = GradScaler() if device == 'cuda' else None  # Initialize the GradScaler for mixed precision if using CUDA

    # Load or define the anatomical prior (B)
    B = get_priors(multi_class=args.multi_class)  # Use multi_class flag from args

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
    def log_metrics(exp_dir, iteration, train_loss, val_loss, mean_gdice, class_gdice, class_hdd, be=None, ts=None, is_best=False):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file = os.path.join(exp_dir, 'logs', 'training_log.txt')
        
        log_entry = f"[{timestamp}] Iteration {iteration}\n"
        log_entry += f"Training Loss: {train_loss:.4f}\n"
        log_entry += f"Validation Loss: {val_loss:.4f}\n"
        log_entry += f"Mean Validation GDice: {mean_gdice:.4f}\n"
        if be is not None and ts is not None:
            log_entry += f"Betti Error: {be}\n"
            log_entry += f"Topological Success: {ts}\n"
        log_entry += "Per-class metrics:\n"
        for i, (dice, hdd) in enumerate(zip(class_gdice, class_hdd)):
            log_entry += f"  Class {i}: GDice={dice:.4f}, HDD={hdd:.4f}\n"
        
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
        all_predictions = []
        all_targets = []
        all_hdd = [[] for _ in range(4)]  # For background and 3 classes
        all_dsc = [[] for _ in range(4)]  # For background and 3 classes
        all_betti_errors = []
        all_topo_success = []
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                if device == 'cuda':
                    with autocast():  # Use autocast for mixed precision
                        outputs = model(images)
                        ce_loss = criterion(outputs, masks)
                else:
                    outputs = model(images.float())
                    ce_loss = criterion(outputs, masks.float())
                total_ce_loss += ce_loss.item()
                
                # Convert predictions to probabilities and then to class labels
                pred = torch.softmax(outputs, dim=1)
                pred_labels = torch.argmax(pred, dim=1)
                pred = pred.cpu().numpy()
                pred_labels = pred_labels.cpu().numpy()
                masks = masks.cpu().numpy()
                
                # Calculate Hausdorff distance and Dice coefficient for each class in the batch
                for class_idx in range(4):  # Including background class
                    for batch_idx in range(pred_labels.shape[0]):
                        pred_binary = (pred_labels[batch_idx] == class_idx).astype(np.float32)
                        true_binary = (masks[batch_idx] == class_idx).astype(np.float32)
                        if np.sum(pred_binary) > 0 and np.sum(true_binary) > 0:  # Only calculate if both masks have the class
                            hdd = hausdorff_distance(pred_binary, true_binary)
                            dsc = dice_coefficient(pred_binary, true_binary)
                            all_hdd[class_idx].append(hdd)
                            all_dsc[class_idx].append(dsc)
                
                # Compute Betti numbers and topological metrics
                betti_pred = compute_class_combinations_betti(pred_labels)
                betti_true = compute_class_combinations_betti(masks)
                be = betti_error(betti_pred, betti_true)
                ts = topological_success(be)
                all_betti_errors.append(be)
                all_topo_success.append(ts)
                
                all_predictions.append(pred)
                all_targets.append(masks)
        
        # Calculate mean CE loss
        mean_ce_loss = total_ce_loss / len(val_loader)
        
        # Concatenate all batches
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Calculate generalized dice score
        all_predictions = np.clip(all_predictions, 1e-7, 1.0)
        all_predictions = all_predictions / all_predictions.sum(axis=1, keepdims=True)
        mean_gdice, class_gdice = generalized_dice(all_predictions, all_targets)
        
        # Calculate mean Hausdorff distance and Dice coefficient for each class
        mean_hdd = [np.mean(class_hdds) if len(class_hdds) > 0 else float('inf') 
                    for class_hdds in all_hdd]
        mean_dsc = [np.mean(class_dscs) if len(class_dscs) > 0 else 0.0 for class_dscs in all_dsc]

        # Calculate mean Betti error and topological success
        mean_betti_error = np.mean(all_betti_errors)
        mean_topo_success = np.mean(all_topo_success)
        return mean_ce_loss, mean_gdice, class_gdice, mean_hdd, mean_dsc, mean_betti_error, mean_topo_success

    # Training loop with argument values
    best_gdice = 0
    iteration = 1
    train_iterator = iter(train_loader)
    pbar = tqdm(total=args.max_iterations, desc='Training')

    while iteration < args.max_iterations:
        model.train()
        
        try:
            images, masks = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            images, masks = next(train_iterator)
        
        images, masks = images.to(device), masks.to(device)
        
        # Forward pass
        if device == 'cuda':
            with autocast():  # Use autocast for mixed precision
                outputs = model(images)
                seg_loss = criterion(outputs, masks)
                loss = seg_loss
        else:
            outputs = model(images.float())
            seg_loss = criterion(outputs, masks.float())
            
            
            loss = seg_loss
        
        # Backward pass
        optimizer.zero_grad()
        if device == 'cuda':
            scaler.scale(loss).backward()  # Scale the loss for mixed precision
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # Store training loss
        train_losses.append(loss.item())
        
        # Update progress bar
        pbar.update(1)
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Evaluate using argument value for validation steps
        if iteration % args.validation_steps == 0:
            val_ce_loss, val_gdice, class_gdice, class_hdd, class_dsc, mean_betti_error, mean_topo_success = evaluate_model(model, val_loader)
            print(f'Iteration {iteration}:')
            print(f'Training CE Loss: {loss.item():.4f}')
            print(f'Validation CE Loss: {val_ce_loss:.4f}')
            print(f'Mean Validation GDice: {val_gdice:.4f}')
            print(f'Betti Error: {mean_betti_error}')
            print(f'Topological Success: {mean_topo_success}')
            print('Per-class Hausdorff distances:')
            for i, hdd in enumerate(class_hdd):
                print(f'  Class {i}: {hdd:.4f}')
            print('Per-class Dice coefficients:')
            for i, dsc in enumerate(class_dsc):
                print(f'  Class {i}: {dsc:.4f}')
            
            # Store metrics
            iterations.append(iteration)
            val_losses.append(val_ce_loss)
            val_gdice_scores.append(val_gdice)
            
            
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
                print(f'New best model saved! Mean GDice: {best_gdice:.4f}')
                # Log the best model metrics
                log_metrics(exp_dir, iteration, loss.item(), val_ce_loss, val_gdice, class_gdice, class_hdd, mean_betti_error, mean_topo_success, is_best=True)
            else:
                # Log regular metrics
                log_metrics(exp_dir, iteration, loss.item(), val_ce_loss, val_gdice, class_gdice, class_hdd, mean_betti_error, mean_topo_success, is_best=False)

        iteration += 1

    pbar.close()
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





import argparse
import os
import torch
import numpy  as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from unet import UNet
from utils.metrics import generalized_dice, dice_coefficient, hausdorff_distance
from utils.dataloading import get_patient_data, ValACDCDataset
from utils.metrics import betti_error, topological_success, compute_percentiles
from utils.metrics import compute_class_combinations_betti

def parse_args():
    parser = argparse.ArgumentParser(description='Test trained UNet model on ACDC dataset')
    parser.add_argument('--test_dir', type=str, default='data/preprocessed/test',
                        help='Directory containing test data')
    parser.add_argument('--experiment_dir', type=str, required=True,
                        help='Directory containing the trained model')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for testing')
    parser.add_argument('--save_visualizations', action='store_true',
                        help='Save segmentation visualizations')
    return parser.parse_args()

def save_segmentation_visualization(image, true_mask, pred_mask, save_path):
    """
    Create and save a visualization of the segmentation results.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot original image
    axes[0].imshow(image[0], cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Plot ground truth
    axes[1].imshow(true_mask, cmap='nipy_spectral')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Plot prediction
    axes[2].imshow(pred_mask, cmap='nipy_spectral')
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def evaluate_model(model, test_loader, device, criterion):
    """Similar to training evaluation but with additional metrics"""
    model.eval()
    total_ce_loss = 0
    all_predictions = []
    all_targets = []
    all_hdd = [[] for _ in range(4)]  # For background and 3 classes
    all_dsc = [[] for _ in range(4)]  # For background and 3 classes
    all_betti = []
    
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            
            # Calculate Cross Entropy Loss
            ce_loss = criterion(outputs, masks)
            total_ce_loss += ce_loss.item()
            
            # Convert predictions
            pred_probs = torch.softmax(outputs, dim=1)  # Shape: (B, C, H, W)
            pred_labels = torch.argmax(pred_probs, dim=1)  # Shape: (B, H, W)
            
            # Convert to numpy
            pred_probs_np = pred_probs.cpu().numpy()  # Shape: (B, C, H, W)
            pred_labels_np = pred_labels.cpu().numpy()  # Shape: (B, H, W)
            masks_np = masks.cpu().numpy()  # Shape: (B, H, W)
            
            # Store for batch metrics
            all_predictions.append(pred_probs_np)  # Storing probabilities for gDSC
            all_targets.append(masks_np)
            
            # Calculate per-class Hausdorff distance and Dice coefficient using pred_labels_np
            for class_idx in range(4):
                for batch_idx in range(pred_labels_np.shape[0]):
                    pred_binary = (pred_labels_np[batch_idx] == class_idx)
                    true_binary = (masks_np[batch_idx] == class_idx)
                    
                    # Use default behavior from metric functions
                    hdd = hausdorff_distance(pred_binary, true_binary)
                    dsc = dice_coefficient(pred_binary, true_binary)
                    all_hdd[class_idx].append(hdd)
                    all_dsc[class_idx].append(dsc)
            
            # Calculate Betti numbers using pred_labels_np
            for i in range(len(masks_np)):
                pred_betti = compute_class_combinations_betti(pred_labels_np[i])
                true_betti = compute_class_combinations_betti(masks_np[i])
                all_betti.append((pred_betti, true_betti))
    
    # Calculate mean metrics
    mean_ce_loss = total_ce_loss / len(test_loader)
    
    # Concatenate predictions and targets
    all_predictions = np.concatenate(all_predictions, axis=0)  # Shape: (N, C, H, W)
    all_targets = np.concatenate(all_targets, axis=0)  # Shape: (N, H, W)
    
    # Calculate generalized dice score with probability predictions
    mean_gdice, class_gdice = generalized_dice(all_predictions, all_targets)
    
    # Calculate mean Hausdorff distance and Dice coefficient per class
    mean_hdd = [np.mean(class_hdds) if len(class_hdds) > 0 else float('inf') 
                for class_hdds in all_hdd]
    mean_dsc = [np.mean(class_dscs) if len(class_dscs) > 0 else 0.0 
                for class_dscs in all_dsc]
    
    # Calculate Betti metrics
    betti_errors = []
    topological_successes = []
    for pred_betti, true_betti in all_betti:
        for combo in pred_betti.keys():
            pred_combo_betti = pred_betti[combo][:2]
            true_combo_betti = true_betti[combo][:2]
            be = betti_error(pred_combo_betti, true_combo_betti)
            ts = topological_success(be)
            betti_errors.append(be)
            topological_successes.append(ts)
    
    # Calculate percentiles for HDD and DSC per class
    hdd_percentiles = [
        compute_percentiles(class_hdds, [25, 50, 75]) if len(class_hdds) > 0 else [None, None, None]
        for class_hdds in all_hdd
    ]
    dsc_percentiles = [
        compute_percentiles(class_dscs, [25, 50, 75]) if len(class_dscs) > 0 else [None, None, None]
        for class_dscs in all_dsc
    ]
    
    # Calculate percentiles for gDSC
    gdsc_percentiles = compute_percentiles([gd for gd in class_gdice], [25, 50, 75]) if any(class_gdice) else [None, None, None]
    
    # Calculate percentiles for Betti Error
    betti_percentiles = compute_percentiles(betti_errors, [98, 99, 100]) if len(betti_errors) > 0 else [None, None, None]
    
    results = {
        'ce_loss': mean_ce_loss,
        'mean_gdice': mean_gdice,
        'class_gdice': class_gdice.tolist(),
        'mean_hdd': mean_hdd,
        'mean_dsc': mean_dsc,
        'mean_betti_error': np.mean(betti_errors) if len(betti_errors) > 0 else None,
        'std_betti_error': np.std(betti_errors) if len(betti_errors) > 0 else None,
        'topological_success_rate': np.mean(topological_successes) if len(topological_successes) > 0 else None,
        'hdd_percentiles': [hp.tolist() for hp in hdd_percentiles],
        'dsc_percentiles': [dp.tolist() for dp in dsc_percentiles],
        'gdsc_percentiles': gdsc_percentiles.tolist(),
        'betti_percentiles': betti_percentiles.tolist(),
    }
    
    return results

def test_model(model, test_loader, device, results_dir, save_visualizations):
    """Simplified test_model using evaluate_model"""
    criterion = torch.nn.CrossEntropyLoss()
    results = evaluate_model(model, test_loader, device, criterion)
    
    # Save visualizations if requested
    if save_visualizations:
        os.makedirs(os.path.join(results_dir, 'visualizations'), exist_ok=True)
        with torch.no_grad():
            for idx, (images, masks) in enumerate(test_loader):
                images = images.to(device)
                outputs = model(images)
                pred_masks = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
                
                for i in range(images.shape[0]):
                    save_path = os.path.join(results_dir, 'visualizations', f'sample_{idx}_{i}.png')
                    save_segmentation_visualization(
                        images[i].cpu().numpy(),
                        masks[i].cpu().numpy(),
                        pred_masks[i].cpu().numpy(),
                        save_path
                    )
    
    # Save Betti numbers analysis
    betti_file = os.path.join(results_dir, 'betti_numbers.txt')
    with open(betti_file, 'w') as f:
        # Write summary statistics
        f.write("\nSummary Statistics:\n")
        f.write(f"Mean Betti Error: {results['mean_betti_error']:.4f}\n")
        f.write(f"Std Betti Error: {results['std_betti_error']:.4f}\n")
        f.write(f"Topological Success Rate: {results['topological_success_rate']:.4f}\n")
    
    return results

def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load test data
    test_data_dict, test_image_paths, test_lbl_paths = get_patient_data(args.test_dir)
    test_dataset = ValACDCDataset(test_image_paths, test_lbl_paths)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Load model
    model_path = os.path.join(args.experiment_dir, 'models', 'best_model.pth')
    checkpoint = torch.load(model_path)
    
    model = UNet(
        in_channels=1,
        n_classes=4,
        depth=5,
        wf=48,
        padding=True,
        batch_norm=True,
        up_mode='upsample'
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create results directory
    results_dir = os.path.join(args.experiment_dir, 'test_results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Test model
    results = test_model(model, test_loader, device, results_dir, args.save_visualizations)
    
    # Save results
    with open(os.path.join(results_dir, 'metrics.txt'), 'w') as f:
        for metric, value in results.items():
            f.write(f'{metric}: {value}\n')
            print(f'{metric}: {value}')

if __name__ == '__main__':
    main()

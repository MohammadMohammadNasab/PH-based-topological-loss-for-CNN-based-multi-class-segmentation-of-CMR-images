import argparse
import os
import torch
import numpy  as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils.CCA import connected_component_analysis
from unet import UNet
from utils.metrics import generalized_dice, dice_coefficient, hausdorff_distance
from utils.dataloading import get_patient_data, ValACDCDataset
from utils.metrics import betti_error, topological_success, compute_percentiles
from utils.metrics import compute_class_combinations_betti
import datetime

CLASS_LABELS = {
    0: 'Background',
    1: 'RV',
    2: 'MY',
    3: 'LV'
}

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
    parser.add_argument('--apply_cca', action='store_true',
                        help='Apply Connected Component Analysis post-processing')
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

def evaluate_model(model, test_loader, device, criterion, apply_cca=False):
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
            
            if apply_cca:
                # Apply CCA to each class separately
                for class_idx in range(1, 4):  # Skip background class
                    for batch_idx in range(pred_labels_np.shape[0]):
                        class_mask = (pred_labels_np[batch_idx] == class_idx)
                        if class_mask.any():
                            processed_mask = connected_component_analysis(class_mask)
                            pred_labels_np[batch_idx][class_mask] = 0  # Clear original
                            pred_labels_np[batch_idx][processed_mask == 1] = class_idx
            
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

def pretty_print_metrics(results):
    """Helper function to format metrics nicely with 3 decimal places"""
    print("\n=== Segmentation Results ===\n")
    
    print("Cross Entropy Loss: {:.3f}".format(results['ce_loss']))
    
    print("\n=== Dice Scores ===")
    print("Mean Generalized Dice Score: {:.3f}".format(results['mean_gdice']))
    print("\nPer-Class Dice Scores:")
    for idx, score in enumerate(results['mean_dsc']):
        print(f"{CLASS_LABELS[idx]}: {score:.3f}")
    
    print("\n=== Hausdorff Distances ===")
    print("Per-Class Hausdorff Distances:")
    for idx, hdd in enumerate(results['mean_hdd']):
        if hdd == float('inf'):
            print(f"{CLASS_LABELS[idx]}: inf")
        else:
            print(f"{CLASS_LABELS[idx]}: {hdd:.3f}")
    
    print("\n=== Percentiles ===")
    for idx, (hdd_p, dsc_p) in enumerate(zip(results['hdd_percentiles'], results['dsc_percentiles'])):
        print(f"\n{CLASS_LABELS[idx]}:")
        print(f"  DSC [25th, 50th, 75th]: [{', '.join(f'{x:.3f}' if x is not None else 'nan' for x in dsc_p)}]")
        print(f"  HDD [25th, 50th, 75th]: [{', '.join(f'{x:.3f}' if x is not None and x != float('inf') else 'inf' for x in hdd_p)}]")
    
    print("\n=== Topological Metrics ===")
    print(f"Mean Betti Error: {results['mean_betti_error']:.3f}")
    print(f"Std Betti Error: {results['std_betti_error']:.3f}")
    print(f"Topological Success Rate: {results['topological_success_rate']:.3f}")
    print(f"Betti Error Percentiles [98th, 99th, 100th]: [{', '.join(f'{x:.3f}' for x in results['betti_percentiles'])}]")

def save_detailed_results(results, results_dir, apply_cca):
    """Save detailed results to a formatted text file by appending"""
    with open(os.path.join(results_dir, 'detailed_metrics.txt'), 'a') as f:
        f.write("\n" + "="*50 + "\n")  # Separator between runs
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=== ACDC Cardiac Segmentation Results ===\n")
        f.write(f"Post-processing: {'CCA' if apply_cca else 'None'}\n\n")
        
        f.write("Cross Entropy Loss: {:.3f}\n\n".format(results['ce_loss']))
        
        f.write("=== Dice Scores ===\n")
        f.write(f"Mean Generalized Dice Score: {results['mean_gdice']:.3f}\n")
        f.write("\nPer-Class Dice Scores:\n")
        for idx, score in enumerate(results['mean_dsc']):
            f.write(f"{CLASS_LABELS[idx]}: {score:.3f}\n")
        
        f.write("\n=== Hausdorff Distances ===\n")
        for idx, hdd in enumerate(results['mean_hdd']):
            if hdd == float('inf'):
                f.write(f"{CLASS_LABELS[idx]}: inf\n")
            else:
                f.write(f"{CLASS_LABELS[idx]}: {hdd:.3f}\n")
        
        f.write("\n=== Statistical Analysis ===\n")
        for idx, (hdd_p, dsc_p) in enumerate(zip(results['hdd_percentiles'], results['dsc_percentiles'])):
            f.write(f"\n{CLASS_LABELS[idx]}:\n")
            f.write(f"  DSC Percentiles [25th, 50th, 75th]: [{', '.join(f'{x:.3f}' if x is not None else 'nan' for x in dsc_p)}]\n")
            f.write(f"  HDD Percentiles [25th, 50th, 75th]: [{', '.join(f'{x:.3f}' if x is not None and x != float('inf') else 'inf' for x in hdd_p)}]\n")
        
        f.write("\n=== Topological Analysis ===\n")
        f.write(f"Mean Betti Error: {results['mean_betti_error']:.3f}\n")
        f.write(f"Std Betti Error: {results['std_betti_error']:.3f}\n")
        f.write(f"Topological Success Rate: {results['topological_success_rate']:.3f}\n")
        f.write(f"Betti Error Percentiles [98th, 99th, 100th]: [{', '.join(f'{x:.3f}' for x in results['betti_percentiles'])}]\n")

def test_model(model, test_loader, device, results_dir, save_visualizations, apply_cca):
    """Modified test_model function with improved reporting"""
    criterion = torch.nn.CrossEntropyLoss()
    results = evaluate_model(model, test_loader, device, criterion, apply_cca)
    
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
    
    # Print and save detailed results
    pretty_print_metrics(results)
    save_detailed_results(results, results_dir, apply_cca)
    
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
    results = test_model(model, test_loader, device, results_dir, 
                        args.save_visualizations, args.apply_cca)
    
    # Save results by appending
    metrics_file = os.path.join(results_dir, 'metrics.txt')
    with open(metrics_file, 'a') as f:
        f.write("\n" + "="*50 + "\n")  # Separator between runs
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Post-processing: {'CCA' if args.apply_cca else 'None'}\n")
        for metric, value in results.items():
            f.write(f'{metric}: {value}\n')
            print(f'{metric}: {value}')

if __name__ == '__main__':
    main()

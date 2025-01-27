import argparse
import os
import torch
import numpy  as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from unet import UNet
from utils.metrics import generalized_dice, dice_coefficient, hausdorff_distance
from utils.dataloading import get_patient_data, ValACDCDataset
from utils.metrics import betti_error, topological_success
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

def test_model(model, test_loader, device, results_dir, save_visualizations):
    """
    Test the model and calculate metrics.
    """
    model.eval()
    dice_scores = []
    hausdorff_distances = []
    gdice_scores = []
    betti_errors = []
    topological_successes = []
    
    # Add lists for Betti numbers
    pred_betti_numbers = []
    true_betti_numbers = []
    
    os.makedirs(os.path.join(results_dir, 'visualizations'), exist_ok=True)
    
    with torch.no_grad():
        for idx, (images, masks) in enumerate(test_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            predictions = torch.softmax(outputs, dim=1)
            pred_masks = torch.argmax(predictions, dim=1)
            
            # Convert to numpy for metric calculation
            pred_np = pred_masks.cpu().numpy()
            true_np = masks.cpu().numpy()
            
            # Calculate metrics for each image in batch
            for i in range(images.shape[0]):
                # Calculate Dice score for each class
                for class_idx in range(1, 4):  # Exclude background
                    pred_class = (pred_np[i] == class_idx)
                    true_class = (true_np[i] == class_idx)
                    
                    if true_class.sum() > 0:  # Only calculate if class exists in ground truth
                        dice = dice_coefficient(pred_class, true_class)
                        hd = hausdorff_distance(pred_class, true_class)
                        
                        dice_scores.append(dice)
                        hausdorff_distances.append(hd)
                
                # Calculate Generalized Dice Score
                pred_one_hot = torch.nn.functional.one_hot(pred_masks[i], 4).float()
                true_one_hot = torch.nn.functional.one_hot(masks[i], 4).float()
                gdice = generalized_dice(
                    pred_one_hot.unsqueeze(0).permute(0, 3, 1, 2),
                    true_one_hot.unsqueeze(0).permute(0, 3, 1, 2)
                )
                gdice_scores.append(gdice)
                
                # Compute Betti numbers for all class combinations
                pred_betti = compute_class_combinations_betti(pred_np[i])
                true_betti = compute_class_combinations_betti(true_np[i])
                
                pred_betti_numbers.append(pred_betti)
                true_betti_numbers.append(true_betti)
                
                # Calculate Betti error and TS for each combination
                for combo in pred_betti.keys():
                    pred_combo_betti = pred_betti[combo][:2]  # Only β₀ and β₁
                    true_combo_betti = true_betti[combo][:2]  # Only β₀ and β₁
                    
                    be = betti_error(pred_combo_betti, true_combo_betti)
                    ts = topological_success(be)
                    
                    betti_errors.append(be)
                    topological_successes.append(ts)
                
                # Save visualization
                if save_visualizations:
                    save_path = os.path.join(results_dir, 'visualizations', f'sample_{idx}_{i}.png')
                    save_segmentation_visualization(
                        images[i].cpu().numpy(),
                        true_np[i],
                        pred_np[i],
                        save_path
                    )
    
    # Calculate final metrics
    results = {
        'mean_dice': np.mean(dice_scores),
        'std_dice': np.std(dice_scores),
        'mean_hausdorff': np.mean(hausdorff_distances),
        'std_hausdorff': np.std(hausdorff_distances),
        'mean_gdice': np.mean(gdice_scores),
        'std_gdice': np.std(gdice_scores),
        'mean_betti_error': np.mean(betti_errors),
        'std_betti_error': np.std(betti_errors),
        'topological_success_rate': np.mean(topological_successes),
        'betti_numbers': {
            'predictions': pred_betti_numbers,
            'ground_truth': true_betti_numbers
        }
    }
    
    # Save detailed Betti numbers
    betti_file = os.path.join(results_dir, 'betti_numbers.txt')
    with open(betti_file, 'w') as f:
        f.write("Betti Numbers Analysis\n")
        f.write("=====================\n\n")
        
        for idx, (pred_betti, true_betti) in enumerate(zip(pred_betti_numbers, true_betti_numbers)):
            f.write(f"\nSample {idx}:\n")
            
            for combo in pred_betti.keys():
                combo_name = f"Class {combo[0]}" if len(combo) == 1 else f"Classes {combo}"
                f.write(f"\n{combo_name}:\n")
                f.write(f"Predicted: {pred_betti[combo][:2]}\n")
                f.write(f"True: {true_betti[combo][:2]}\n")
            
            f.write("\n" + "-"*50 + "\n")
        
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

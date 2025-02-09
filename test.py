import argparse
import os
import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from CCA import connected_component_analysis
from unet import UNet
from metrics import compute_betti_numbers, compute_class_combinations_betti, generalized_dice, dice_coefficient, hausdorff_distance, betti_error, topological_success
from utils.dataloading import get_patient_data, ValACDCDataset
from topo import multi_class_topological_post_processing
import datetime

CLASS_LABELS = {0: 'Background', 1: 'RV', 2: 'MY', 3: 'LV'}

# **Parse Arguments**
def parse_args():
    parser = argparse.ArgumentParser(description='Test trained UNet model on ACDC dataset')
    parser.add_argument('--test_dir', type=str, default='data/preprocessed/test', help='Directory containing test data')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model .pth file')
    parser.add_argument('--results_dir', type=str, default='test_results', help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for testing')
    parser.add_argument('--save_visualizations', action='store_true', help='Save segmentation visualizations')
    parser.add_argument('--apply_cca', action='store_true', help='Apply Connected Component Analysis post-processing')
    parser.add_argument('--apply_topo', action='store_true', help='Apply Persistent Homology post-processing')
    parser.add_argument('--multi_class', action='store_true', help='Use Multi-Class Priors for Topological Post-Processing')
    parser.add_argument('--description', type=str, default='', help='Description of the test run')
    return parser.parse_args()

# **Set Priors Based on `--multi_class` Argument**
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

def compute_percentiles(data, percentiles=[25, 50, 75]):
    """Compute percentiles for a list of values."""
    if len(data) == 0:
        return [None] * len(percentiles)  # Return None if data is empty
    return np.percentile(data, percentiles).tolist()

def plot_losses(all_losses, save_path):
    """Plot topo and MSE losses for all samples."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot topological losses
    for i, losses in enumerate(all_losses):
        ax1.plot(losses['topo_loss'], label=f'Sample {i+1}', alpha=0.7)
    ax1.set_title('Topological Loss')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # Plot MSE losses
    for i, losses in enumerate(all_losses):
        ax2.plot(losses['mse_loss'], label=f'Sample {i+1}', alpha=0.7)
    ax2.set_title('MSE Loss')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')
    ax2.grid(True)
    
    # Add legend to the right of the figure
    fig.legend(bbox_to_anchor=(1.15, 0.5), loc='center left')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

# **Updated Evaluate Model Function**
def evaluate_model(model, test_loader, device, criterion, apply_cca=False, apply_topo=False, multi_class=False):
    model.eval()
    print(device)

    total_ce_loss = 0
    all_predictions = []
    all_targets = []
    all_hdd = [[] for _ in range(4)]  
    all_dsc = [[] for _ in range(4)]  
    all_gdsc = []  # Add this line to store generalized dice scores
    all_betti = []
    all_topological_success = []  

    priors = get_priors(multi_class)

    all_losses = []  # Track losses for all samples

    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm.tqdm(test_loader, total=len(test_loader))):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            # Calculate Cross Entropy Loss
            ce_loss = criterion(outputs, masks)
            total_ce_loss += ce_loss.item()

            # Convert predictions
            pred_probs = torch.softmax(outputs, dim=1)
            pred_labels = torch.argmax(pred_probs, dim=1)

            # Calculate generalized dice score
            gdsc, _ = generalized_dice(pred_probs, masks)
            all_gdsc.append(gdsc)

            # Convert to numpy
            pred_probs_np = pred_probs.cpu().numpy()
            pred_labels_np = pred_labels.cpu().numpy()
            masks_np = masks.cpu().numpy()

            # **Apply CCA (Connected Component Analysis)**
            for class_idx in range(1, 4):  # Skip background
                for batch_idx in range(pred_labels_np.shape[0]):
                    class_mask = (pred_labels_np[batch_idx] == class_idx)
                    if class_mask.any():
                        processed_mask = connected_component_analysis(class_mask, connectivity=1)
                        pred_labels_np[batch_idx][class_mask] = 0  # Clear original
                        pred_labels_np[batch_idx][processed_mask == 1] = class_idx

            # **Apply Persistent Homology Post-Processing**
            if apply_topo:
                with torch.enable_grad():
                    model.train()  # Set model to training mode for topological post-processing
                    print(f"Applying Persistent Homology Post-Processing with {'Multi-Class' if multi_class else 'Single-Class'} Priors...")
                    processed_outputs = []
                    batch_losses = []
                    
                    for j in range(images.shape[0]):
                        input_single = images[j].unsqueeze(0)
                        topo_model, losses = multi_class_topological_post_processing(
                            input_single, model, priors, lr=1e-5, mse_lambda=1000,
                            num_its=100, thresh=0.5, parallel=False
                        )
                        refined_output = topo_model(input_single)
                        processed_outputs.append(refined_output)
                        all_losses.append(losses)

                    outputs = torch.cat(processed_outputs, dim=0)

                pred_probs = torch.softmax(outputs, dim=1)  # Updated predictions
                pred_labels = torch.argmax(pred_probs, dim=1)  # Updated predictions
                pred_labels_np = pred_labels.cpu().numpy()
            model.eval()
            # Store results
            all_predictions.append(pred_probs_np)
            all_targets.append(masks_np)

            # Compute metrics
            for class_idx in range(4):  # Compute per-class HDD and DSC
                for batch_idx in range(pred_labels_np.shape[0]):
                    pred_binary = (pred_labels_np[batch_idx] == class_idx)
                    true_binary = (masks_np[batch_idx] == class_idx)
                    all_hdd[class_idx].append(hausdorff_distance(pred_binary, true_binary))
                    all_dsc[class_idx].append(dice_coefficient(pred_binary, true_binary))

            # Compute Betti Errors & Topological Success Rate
            for i in range(len(masks_np)):
                pred_betti_numbers = compute_class_combinations_betti(pred_labels_np[i])
                true_betti_numbers = compute_class_combinations_betti(masks_np[i])

                pred_betti = betti_error(pred_betti_numbers, true_betti_numbers)
                all_betti.append(pred_betti)
                all_topological_success.append(topological_success(pred_betti))  # Compute TSR

    # **Compute Percentiles**
    hdd_percentiles = [compute_percentiles(class_hdds, [25, 50, 75]) for class_hdds in all_hdd]
    dsc_percentiles = [compute_percentiles(class_dscs, [25, 50, 75]) for class_dscs in all_dsc]
    betti_percentiles = compute_percentiles(all_betti, [98, 99, 100])
    # Compute Topological Success Rate Mean & STD
    mean_tsr = np.mean(all_topological_success)
    std_tsr = np.std(all_topological_success)

    # Update results dictionary to include generalized dice metrics
    results = {
        'ce_loss': total_ce_loss / len(test_loader),
        'gdsc_mean': np.mean(all_gdsc),
        'gdsc_std': np.std(all_gdsc),
        'hdd_percentiles': hdd_percentiles,
        'dsc_percentiles': dsc_percentiles,
        'betti_error_percentiles': betti_percentiles,
        'topological_success_rate': mean_tsr,
        'std_topological_success_rate': std_tsr,
        'all_losses': all_losses
    }

    return results

# **Test Model**
def test_model(model, test_loader, device, results_dir, save_visualizations, apply_cca, apply_topo, multi_class, description):
    criterion = torch.nn.CrossEntropyLoss()
    results = evaluate_model(model, test_loader, device, criterion, apply_cca, apply_topo, multi_class)
    # Plot losses if topological post-processing was applied
    all_losses = results['all_losses']
    if apply_topo and all_losses:
        plot_losses(all_losses, os.path.join(results_dir, f'loss_trends_{"multi" if multi_class else "single"}_class.png'))

    # Print results
    print("\n=== Test Results ===")
    for key, value in results.items():
        print(f"{key}: {value}")

    # Save results
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'metrics.txt'), 'a') as f:
        f.write("\n" + "="*50 + "\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Description: {description}\n")
        f.write(f"Post-processing: {'CCA' if apply_cca else 'None'}, {'Topology' if apply_topo else 'None'}\n")
        f.write(f"Multi-Class: {multi_class}\n\n")

        # Log results with percentile order
        for key, value in results.items():
            if "percentiles" in key:
                if "hdd" in key or "dsc" in key:
                    f.write(f"{key} (Ordered as [25th, 50th, 75th]): {value}\n")
                elif "betti_error" in key:
                    f.write(f"{key} (Ordered as [98th, 99th, 100th]): {value}\n")
            else:
                if key != 'all_losses':
                    f.write(f"{key}: {value}\n")

    return results

# **Main Execution**
def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    _, test_image_paths, test_lbl_paths = get_patient_data(args.test_dir)
    test_dataset = ValACDCDataset(test_image_paths, test_lbl_paths)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    checkpoint = torch.load(args.model_path)
    model = UNet(in_channels=1, n_classes=4, depth=5, wf=48, padding=True, batch_norm=True, up_mode='upsample').to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_model(model, test_loader, device, args.results_dir, args.save_visualizations, args.apply_cca, args.apply_topo, args.multi_class, args.description)

if __name__ == '__main__':
    main()

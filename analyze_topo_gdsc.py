import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from scipy import stats
from tqdm import tqdm

from utils.dataloading import get_patient_data, ValACDCDataset
from unet import UNet
from metrics import generalized_dice
from topo import compute_topological_loss

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze relationship between topological loss and gDSC')
    parser.add_argument('--test_dir', type=str, default='data/preprocessed/test')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--results_dir', type=str, default='analysis_results')
    parser.add_argument('--multi_class', action='store_true')
    return parser.parse_args()

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
            (1,): (1, 0),
            (2,): (1, 1),
            (3,): (1, 0),
        }

def normalize_metric(values):
    """Min-max normalization of values."""
    min_val = np.min(values)
    max_val = np.max(values)
    if max_val == min_val:
        return np.zeros_like(values)
    return (values - min_val) / (max_val - min_val)

def compute_metrics(model, test_loader, device, priors):
    model.eval()
    topo_losses = []
    gdsc_scores = []
    skipped_samples = 0
    
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Computing metrics"):
            try:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                pred_probs = torch.softmax(outputs, dim=1)
                
                # Compute gDSC
                gdsc, _ = generalized_dice(pred_probs, masks)
                
                # Compute topological loss
                topo_loss = compute_topological_loss(
                    torch.softmax(outputs, 1).detach().squeeze(0),
                    priors)
                gdsc_scores.append(gdsc)
                topo_losses.append(topo_loss.item())
            except Exception as e:
                print(f"Skipping sample due to error: {str(e)}")
                skipped_samples += 1
                continue
    
    if skipped_samples > 0:
        print(f"Skipped {skipped_samples} samples due to errors")
    
    return np.array(topo_losses), np.array(gdsc_scores)

def plot_relationship(topo_losses, gdsc_scores, save_path):
    # Normalize metrics
    norm_topo = normalize_metric(topo_losses)
    norm_gdsc = normalize_metric(gdsc_scores)
    
    # Compute correlation coefficient and p-value
    rho, p_value = stats.spearmanr(norm_topo, norm_gdsc)
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(norm_topo, norm_gdsc, alpha=0.5)
    plt.xlabel('Normalized Topological Loss')
    plt.ylabel('Normalized gDSC')
    plt.title(f'Relationship between Topological Loss and gDSC\nSpearman ρ = {rho:.3f} (p = {p_value:.3e})')
    
    # Add trend line
    z = np.polyfit(norm_topo, norm_gdsc, 1)
    p = np.poly1d(z)
    plt.plot(norm_topo, p(norm_topo), "r--", alpha=0.8)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig(save_path)
    plt.close()
    
    return rho, p_value

def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Load test data
    _, test_image_paths, test_lbl_paths = get_patient_data(args.test_dir)
    test_dataset = ValACDCDataset(test_image_paths, test_lbl_paths)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Load model
    checkpoint = torch.load(args.model_path)
    model = UNet(in_channels=1, n_classes=4, depth=5, wf=48, 
                padding=True, batch_norm=True, up_mode='upsample').to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Get priors
    priors = get_priors(args.multi_class)
    
    # Compute metrics
    topo_losses, gdsc_scores = compute_metrics(model, test_loader, device, priors)
    
    # Plot and save results
    save_path = os.path.join(args.results_dir, 'topo_gdsc_relationship.png')
    rho, p_value = plot_relationship(topo_losses, gdsc_scores, save_path)
    
    # Save numerical results
    results_path = os.path.join(args.results_dir, 'correlation_results.txt')
    with open(results_path, 'w') as f:
        f.write(f"Analysis Results:\n")
        f.write(f"================\n")
        f.write(f"Number of samples: {len(topo_losses)}\n")
        f.write(f"Spearman correlation coefficient (ρ): {rho:.3f}\n")
        f.write(f"P-value: {p_value:.3e}\n")
        f.write(f"\nTopological Loss stats:\n")
        f.write(f"Mean: {np.mean(topo_losses):.3f}\n")
        f.write(f"Std: {np.std(topo_losses):.3f}\n")
        f.write(f"\ngDSC stats:\n")
        f.write(f"Mean: {np.mean(gdsc_scores):.3f}\n")
        f.write(f"Std: {np.std(gdsc_scores):.3f}\n")
    
    print(f"Results saved to {args.results_dir}")

if __name__ == '__main__':
    main()

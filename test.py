import argparse
import os

import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from CCA import connected_component_analysis
from unet import UNet
from utils.metrics import generalized_dice, dice_coefficient, hausdorff_distance, betti_error, topological_success
from utils.dataloading import get_patient_data, ValACDCDataset
from topo import multi_class_topological_post_processing
import datetime

CLASS_LABELS = {0: 'Background', 1: 'RV', 2: 'MY', 3: 'LV'}


MULTI_CLASS = True
if MULTI_CLASS:
    prior = {
        (1,):   (1, 0),
        (2,):   (1, 1),
        (3,):   (1, 0),
        (1, 2): (1, 1),
        (1, 3): (2, 0),
        (2, 3): (1, 0)
    }
else:
    prior = {
        (1,):   (1, 0),
        (2,):   (1, 1),
        (3,):   (1, 0),
    }
def parse_args():
    parser = argparse.ArgumentParser(description='Test trained UNet model on ACDC dataset')
    parser.add_argument('--test_dir', type=str, default='data/preprocessed/test', help='Directory containing test data')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model .pth file')
    parser.add_argument('--results_dir', type=str, default='test_results', help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for testing')
    parser.add_argument('--save_visualizations', action='store_true', help='Save segmentation visualizations')
    parser.add_argument('--apply_cca', action='store_true', help='Apply Connected Component Analysis post-processing')
    parser.add_argument('--apply_topo', action='store_true', help='Apply Persistent Homology post-processing')
    parser.add_argument('--description', type=str, default='', help='Description of the test run')
    return parser.parse_args()

def evaluate_model(model, test_loader, device, criterion, apply_cca=False, apply_topo=False):
    """Evaluates the model with additional post-processing options."""
    model.eval()
    print(device)
    total_ce_loss = 0
    all_predictions = []
    all_targets = []
    all_hdd = [[] for _ in range(4)]
    all_dsc = [[] for _ in range(4)]
    all_betti = []

    with torch.no_grad():
        for images, masks in tqdm.tqdm(test_loader, total=len(test_loader)):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            # Calculate Cross Entropy Loss
            ce_loss = criterion(outputs, masks)
            total_ce_loss += ce_loss.item()

            # Convert predictions
            pred_probs = torch.softmax(outputs, dim=1)  # Shape: (B, C, H, W)
            pred_labels = torch.argmax(pred_probs, dim=1)  # Shape: (B, H, W)

            # Convert to numpy
            pred_probs_np = pred_probs.cpu().numpy()
            pred_labels_np = pred_labels.cpu().numpy()
            masks_np = masks.cpu().numpy()

            # **Apply CCA (Connected Component Analysis)**
            if apply_cca:
                for class_idx in range(1, 4):
                    for batch_idx in range(pred_labels_np.shape[0]):
                        class_mask = (pred_labels_np[batch_idx] == class_idx)
                        if class_mask.any():
                            processed_mask = connected_component_analysis(class_mask)
                            pred_labels_np[batch_idx][class_mask] = 0  # Clear original
                            pred_labels_np[batch_idx][processed_mask == 1] = class_idx

            # **Apply Persistent Homology Post-Processing**
            if apply_topo:
                print("Applying Persistent Homology Post-Processing...")
                processed_outputs = []
                for i in range(images.shape[0]):  # Process each image separately
                    input_single = images[i].unsqueeze(0).to(device)

                    refined_output = multi_class_topological_post_processing(
                        input_single, model, prior, lr=1e-5, mse_lambda=1.0
                    )

                    processed_outputs.append(refined_output.to(device))

                outputs = torch.cat(processed_outputs, dim=0).to(device)

                pred_probs = torch.softmax(outputs, dim=1)  # Updated predictions
                pred_labels = torch.argmax(pred_probs, dim=1)  # Updated predictions
                pred_labels_np = pred_labels.cpu().numpy()

            # Store results
            all_predictions.append(pred_probs_np)
            all_targets.append(masks_np)

            # Compute metrics
            for class_idx in range(4):
                for batch_idx in range(pred_labels_np.shape[0]):
                    pred_binary = (pred_labels_np[batch_idx] == class_idx)
                    true_binary = (masks_np[batch_idx] == class_idx)
                    all_hdd[class_idx].append(hausdorff_distance(pred_binary, true_binary))
                    all_dsc[class_idx].append(dice_coefficient(pred_binary, true_binary))

            for i in range(len(masks_np)):
                pred_betti = betti_error(pred_labels_np[i], masks_np[i])
                all_betti.append(pred_betti)

    mean_ce_loss = total_ce_loss / len(test_loader)
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    mean_gdice, class_gdice = generalized_dice(all_predictions, all_targets)
    mean_hdd = [np.mean(class_hdds) for class_hdds in all_hdd]
    mean_dsc = [np.mean(class_dscs) for class_dscs in all_dsc]

    results = {
        'ce_loss': mean_ce_loss,
        'mean_gdice': mean_gdice,
        'mean_hdd': mean_hdd,
        'mean_dsc': mean_dsc,
        'mean_betti_error': np.mean(all_betti),
    }

    return results

def test_model(model, test_loader, device, results_dir, save_visualizations, apply_cca, apply_topo, description):
    criterion = torch.nn.CrossEntropyLoss()
    results = evaluate_model(model, test_loader, device, criterion, apply_cca, apply_topo)

    # Print results
    print("\n=== Test Results ===")
    for key, value in results.items():
        print(f"{key}: {value}")

    # Save results
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'metrics.txt'), 'a') as f:
        f.write("\n" + "="*50 + "\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model}\n")
        f.write(f"Post-processing: {'CCA' if apply_cca else 'None'}, {'Topology' if apply_topo else 'None'}\n")
        for key, value in results.items():
            f.write(f"{key}: {value}\n")

    return results

def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    test_data_dict, test_image_paths, test_lbl_paths = get_patient_data(args.test_dir)
    test_dataset = ValACDCDataset(test_image_paths, test_lbl_paths)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    checkpoint = torch.load(args.model_path)
    model = UNet(in_channels=1, n_classes=4, depth=5, wf=48, padding=True, batch_norm=True, up_mode='upsample').to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_model(model, test_loader, device, args.results_dir, args.save_visualizations, args.apply_cca, args.apply_topo, args.description)

if __name__ == '__main__':
    main()

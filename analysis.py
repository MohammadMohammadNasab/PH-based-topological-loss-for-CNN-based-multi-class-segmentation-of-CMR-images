import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import torch
from torch.utils.data import DataLoader
from utils.dataloading import TopoACDCDataset,  get_patient_data
from unet import UNet
from utils.metrics import generalized_dice, betti_error, topological_success
from utils.topo import multi_class_topological_post_processing

def normalize_metric(values):
    # Min-max normalization
    min_val, max_val = np.min(values), np.max(values)
    if max_val == min_val:  # Avoid division by zero
        return np.zeros_like(values)
    return (values - min_val) / (max_val - min_val)

def analyze_topo_spatial_relationship(betti_errors, topo_success, gdsc_scores):
    """
    betti_errors: array-like of Betti error values for each sample
    topo_success: array-like of topological success (0 or 1) for each sample
    gdsc_scores:  array-like of gDSC (or DSC) scores for each sample
    """
    # Convert to numpy
    be_arr = np.array(betti_errors, dtype=float)
    ts_arr = np.array(topo_success, dtype=float)
    gdsc_arr = np.array(gdsc_scores, dtype=float)

    # Define L_topo as some function of betti errors (e.g., L_topo = betti_error)
    l_topo_arr = be_arr.copy()

    # Normalize metrics
    be_norm = normalize_metric(be_arr)
    gdsc_norm = normalize_metric(gdsc_arr)
    l_topo_norm = normalize_metric(l_topo_arr)

    # Compute correlations
    pearson_corr_gdsc, _ = pearsonr(l_topo_arr, gdsc_arr)
    spearman_corr_gdsc, _ = spearmanr(l_topo_arr, gdsc_arr)
    pearson_corr_be, _ = pearsonr(ts_arr, be_arr)
    spearman_corr_be, _ = spearmanr(ts_arr, be_arr)

    print(f"Pearson correlation between L_topo & gDSC: {pearson_corr_gdsc:.3f}")
    print(f"Spearman correlation between L_topo & gDSC: {spearman_corr_gdsc:.3f}")
    print(f"Pearson correlation between TS & Betti Error: {pearson_corr_be:.3f}")
    print(f"Spearman correlation between TS & Betti Error: {spearman_corr_be:.3f}")

    # Scatter plot: Normalized L_topo vs. Normalized gDSC
    plt.figure(figsize=(6,5))
    for i, ts in enumerate(ts_arr):
        color = 'blue' if ts == 1 else 'red'
        plt.scatter(l_topo_norm[i], gdsc_norm[i], c=color, alpha=0.6)
    plt.title("Normalized L_topo vs. Normalized gDSC")
    plt.xlabel("L_topo (normalized)")
    plt.ylabel("gDSC (normalized)")
    plt.grid(True)
    plt.show()

    # Scatter plot: Normalized L_topo vs. Betti Error
    plt.figure(figsize=(6,5))
    for i, ts in enumerate(ts_arr):
        color = 'blue' if ts == 1 else 'red'
        plt.scatter(l_topo_norm[i], be_arr[i], c=color, alpha=0.6)
    plt.title("Normalized L_topo vs. Betti Error")
    plt.xlabel("L_topo (normalized)")
    plt.ylabel("Betti Error")
    plt.grid(True)
    plt.show()

def analyze_extended_metrics(gdsc_values, loss_topo_values):
    """
    Compute correlation between gDSC and topological loss (loss_topo) for each sample,
    normalize both, then plot the relationship.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr, spearmanr

    def normalize_metric(values):
        # ...existing code...
        min_val, max_val = np.min(values), np.max(values)
        if max_val == min_val:
            return np.zeros_like(values)
        return (values - min_val) / (max_val - min_val)

    gdsc_arr = np.array(gdsc_values, dtype=float)
    topo_loss_arr = np.array(loss_topo_values, dtype=float)

    # Normalize both
    gdsc_norm = normalize_metric(gdsc_arr)
    topo_loss_norm = normalize_metric(topo_loss_arr)

    # Compute correlations
    pearson_corr, _ = pearsonr(gdsc_arr, topo_loss_arr)
    spearman_corr, _ = spearmanr(gdsc_arr, topo_loss_arr)
    print(f"Pearson correlation (gDSC vs. L_topo): {pearson_corr:.3f}")
    print(f"Spearman correlation (gDSC vs. L_topo): {spearman_corr:.3f}")

    # Scatter plot
    plt.figure(figsize=(6,5))
    plt.scatter(topo_loss_norm, gdsc_norm, c='blue', alpha=0.7)
    plt.title("Normalized L_topo vs. Normalized gDSC")
    plt.xlabel("L_topo (normalized)")
    plt.ylabel("gDSC (normalized)")
    plt.grid(True)
    plt.show()

def analyze_gdsc_topo_loss(model, data_loader, device, prior):
    """
    For each sample, compute gDSC and topological loss (L_topo) by calling
    multi_class_topological_post_processing. Then normalize and plot.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr, spearmanr

    def normalize_metric(values):
        min_val, max_val = np.min(values), np.max(values)
        if max_val == min_val:
            return np.zeros_like(values)
        return (values - min_val) / (max_val - min_val)

    model.eval()
    gdsc_scores, topo_losses = [], []

    with torch.no_grad():
        for images, masks in data_loader:
            images, masks = images.to(device), masks.to(device)

            # Compute gDSC
            outputs = model(images)
            pred_probs = torch.softmax(outputs, dim=1).cpu().numpy()
            masks_np = masks.cpu().numpy()
            gdsc_mean, _ = generalized_dice(pred_probs, masks_np)
            gdsc_scores.append(gdsc_mean)

            # Derive topological loss from multi_class_topological_post_processing
            # Assuming we modify it to return (model_topo, topological_loss)
            _, loss_topo_val = multi_class_topological_post_processing(
                inputs=images, model=model, prior=prior,
                lr=0.001, mse_lambda=1000,
                opt=torch.optim.Adam, num_its=100, construction='0', thresh=None, parallel=False
            )
            # Store the computed topological loss
            topo_losses.append(loss_topo_val)

    # Normalize
    gdsc_arr = np.array(gdsc_scores)
    topo_arr = np.array(topo_losses)
    gdsc_norm = normalize_metric(gdsc_arr)
    topo_norm = normalize_metric(topo_arr)

    # Compute correlations
    pearson_corr, _ = pearsonr(gdsc_arr, topo_arr)
    spearman_corr, _ = spearmanr(gdsc_arr, topo_arr)
    print(f"Pearson correlation (gDSC vs. L_topo): {pearson_corr:.3f}")
    print(f"Spearman correlation (gDSC vs. L_topo): {spearman_corr:.3f}")

    # Scatter plot
    plt.figure(figsize=(6,5))
    plt.scatter(topo_norm, gdsc_norm, c='blue', alpha=0.7)
    plt.title("Normalized L_topo vs. Normalized gDSC")
    plt.xlabel("L_topo (normalized)")
    plt.ylabel("gDSC (normalized)")
    plt.grid(True)
    plt.show()

def run_analysis_with_model(model, data_loader, device):
    """
    Loads a model, runs predictions on the data_loader, gathers
    Betti errors, topological success, and gDSC, then calls
    analyze_topo_spatial_relationship.
    """
    model.eval()
    betti_errors = []
    topo_successes = []
    gdsc_scores = []
    with torch.no_grad():
        for images, masks in data_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)  # shape (B, C, H, W)
            pred_probs = torch.softmax(outputs, dim=1).cpu().numpy()
            masks_np = masks.cpu().numpy()

            # Compute gDSC
            # (Assuming a single batch. For multiple samples, loop or reshape as needed.)
            gdsc_mean, _ = generalized_dice(pred_probs, masks_np)
            gdsc_scores.append(gdsc_mean)

            # Compute Betti error (example for single predicted class or combination)
            # This part depends on how your classes are structured:
            # Here, assume each sample is binary or a single class for demonstration.
            # Replace with actual logic for multi-class Betti calculations if needed.
            pred_binary = (pred_probs.argmax(axis=1) > 0).astype(int)
            true_binary = (masks_np > 0).astype(int)
            be = betti_error(pred_binary, true_binary)  # updated with real betti calculation
            ts = topological_success(be)

            betti_errors.append(be)
            topo_successes.append(ts)

    # Call existing function to analyze relationship
    analyze_topo_spatial_relationship(betti_errors, topo_successes, gdsc_scores)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model_path)
    
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
    _, image_paths, label_paths = get_patient_data(r'C:\Users\Mohammad\Desktop\ML_Project\data\preprocessed\test')
    dataset = TopoACDCDataset(image_paths, label_paths)
    dataloder = DataLoader(dataset, 1, shuffle=False)
    prior = {
        (1,):   (1, 0),
        (2,):   (1, 1),
        (3,):   (1, 0),
        (1, 2): (1, 1),
        (1, 3): (2, 0),
        (2, 3): (1, 0)
    }
    analyze_gdsc_topo_loss(model, dataloder, device= device, prior=prior)
if __name__ == "__main__":
    main()
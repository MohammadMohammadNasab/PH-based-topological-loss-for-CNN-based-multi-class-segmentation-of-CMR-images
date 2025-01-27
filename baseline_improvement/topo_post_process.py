import torch
import torch.nn.functional as F
import numpy as np
from multiprocessing import Pool
import cripser as crip 
import tcripser as trip
import copy

def get_roi(X, thresh=0.01):
    """Get region of interest from tensor where values exceed threshold"""
    true_points = torch.nonzero(X >= thresh) 
    corner1 = true_points.min(dim=0)[0]
    corner2 = true_points.max(dim=0)[0]
    roi = [slice(None, None)] + [slice(c1, c2 + 1) for c1, c2 in zip(corner1, corner2)]
    return roi

def get_differentiable_barcode(tensor, barcode):
    """Make CubicalRipser barcode differentiable using PyTorch"""
    # Find infinite persistence components
    inf = barcode[barcode[:, 2] == np.finfo(barcode.dtype).max]
    fin = barcode[barcode[:, 2] < np.finfo(barcode.dtype).max]
    
    # Get birth of infinite features
    inf_birth = tensor[tuple(inf[:, 3:3+tensor.ndim].astype(np.int64).T)]
    
    # Calculate persistence of finite features
    births = tensor[tuple(fin[:, 3:3+tensor.ndim].astype(np.int64).T)]
    deaths = tensor[tuple(fin[:, 6:6+tensor.ndim].astype(np.int64).T)]
    delta_p = deaths - births
    
    # Split and sort by dimension
    delta_p = [delta_p[fin[:, 0] == d] for d in range(tensor.ndim)]
    delta_p = [torch.sort(d, descending=True)[0] for d in delta_p]
    
    return inf_birth, delta_p

def topo_post_process(inputs, model, prior, 
                     lr=0.001,
                     mse_lambda=1000.0,
                     num_iters=100, 
                     thresh=0.01,
                     parallel=True):
    """
    Perform topological post-processing on network predictions
    
    Args:
        inputs: Input tensor [1, num_classes, H, W] 
        model: Trained segmentation model
        prior: Dict mapping class indices to target Betti numbers
        lr: Learning rate
        mse_lambda: MSE loss weight
        num_iters: Number of optimization iterations
        thresh: Threshold for ROI
        parallel: Whether to parallelize persistence computation
    
    Returns:
        Post-processed model with improved topological properties
    """
    
    # Get device and dimensions
    device = inputs.device
    spatial_dims = list(inputs.shape[2:])
    
    # Get initial prediction
    model.eval()
    with torch.no_grad():
        pred_orig = torch.softmax(model(inputs), 1).detach().squeeze()
        
    # Get ROI if threshold provided
    if thresh:
        roi = get_roi(pred_orig[1:].sum(0).squeeze(), thresh)
    else:
        roi = [slice(None, None)] * (len(spatial_dims) + 1)
        
    # Initialize post-processed model and optimizer    
    model_topo = copy.deepcopy(model)
    model_topo.eval()
    optimizer = torch.optim.Adam(model_topo.parameters(), lr=lr)
    
    # Convert prior to tensors
    max_dims = [len(b) for b in prior.values()]
    prior = {torch.tensor(c): torch.tensor(b) for c, b in prior.items()}
    
    # Get appropriate persistence computation function
    PH = {'0': crip.computePH, 'N': trip.computePH}
    
    # Main optimization loop
    for it in range(num_iters):
        
        optimizer.zero_grad()
        
        # Get current prediction
        outputs = torch.softmax(model_topo(inputs), 1).squeeze()
        outputs_roi = outputs[roi]
        
        # Compute class-wise persistence
        combos = torch.stack([outputs_roi[c.T].sum(0) for c in prior.keys()])
        combos = 1 - combos # Invert for sub-level persistence
        
        # Compute barcodes
        combos_arr = combos.detach().cpu().numpy().astype(np.float64)
        if parallel:
            with Pool(len(prior)) as p:
                bcodes_arr = p.starmap(PH['0'], zip(combos_arr, max_dims))
        else:
            bcodes_arr = [PH['0'](combo, d) for combo, d in zip(combos_arr, max_dims)]
            
        # Make barcodes differentiable
        max_features = max([b.shape[0] for b in bcodes_arr])
        bcodes = torch.zeros([len(prior), max(max_dims), max_features], device=device)
        
        for c, (combo, bcode) in enumerate(zip(combos, bcodes_arr)):
            _, fin = get_differentiable_barcode(combo, bcode)
            for dim in range(len(spatial_dims)):
                bcodes[c, dim, :len(fin[dim])] = fin[dim]
                
        # Match features to prior
        stacked_prior = torch.stack(list(prior.values()))
        stacked_prior.T[0] -= 1  # Account for infinite 0D component
        matching = torch.zeros_like(bcodes).bool()
        
        for c, combo in enumerate(stacked_prior):
            for dim in range(len(combo)):
                matching[c, dim, :combo[dim]] = True
                
        # Compute losses    
        topo_match = (1 - bcodes[matching]).sum()  # Matching components
        topo_violation = bcodes[~matching].sum()  # Violating components
        mse = F.mse_loss(outputs, pred_orig)  # MSE with original prediction
        
        # Total loss and optimization step
        loss = topo_match + topo_violation + mse_lambda * mse
        loss.backward()
        optimizer.step()
        
    return model_topo

# Example usage:
if __name__ == "__main__":
    # Dummy example
    inputs = torch.randn(1, 3, 64, 64)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(16, 2, 1)
    )
    
    prior = {
        (0,): (1, 0),  # Class 0 should have 1 component, 0 holes
        (1,): (2, 1)   # Class 1 should have 2 components, 1 hole
    }
    
    model_topo = topo_post_process(inputs, model, prior)

from multiprocessing import Pool
import cripser as crip
import tcripser as trip
import numpy as np
import torch
import torch.nn.functional as F
import copy
from torch.optim import SGD, Adam

def crip_wrapper(X, D):
    return crip.computePH(X, maxdim=D)

def trip_wrapper(X, D):
    return trip.computePH(X, maxdim=D)

def get_roi(X, thresh=0.01):
    """Finds the region of interest (ROI) in the segmentation output."""
    true_points = torch.nonzero(X >= thresh)
    if len(true_points) == 0:
        return [slice(None, None)] * (X.dim() + 1)  # Return full volume if nothing detected
    
    corner1 = true_points.min(dim=0)[0]
    corner2 = true_points.max(dim=0)[0]
    
    roi = [slice(None, None)] + [slice(c1, c2 + 1) for c1, c2 in zip(corner1, corner2)]
    return roi

def get_differentiable_barcode(tensor, barcode):
    """Convert persistent homology barcode into a differentiable form for PyTorch."""
    inf = barcode[barcode[:, 2] == np.finfo(barcode.dtype).max]
    fin = barcode[barcode[:, 2] < np.finfo(barcode.dtype).max]
    
    inf_birth = tensor[tuple(inf[:, 3:3+tensor.ndim].astype(np.int64).T)]
    
    births = tensor[tuple(fin[:, 3:3+tensor.ndim].astype(np.int64).T)]
    deaths = tensor[tuple(fin[:, 6:6+tensor.ndim].astype(np.int64).T)]
    delta_p = (deaths - births)
    
    delta_p = [delta_p[fin[:, 0] == d] for d in range(tensor.ndim)]
    delta_p = [torch.sort(d, descending=True)[0] for d in delta_p]
    
    return inf_birth, delta_p

def multi_class_topological_post_processing(
    inputs, model, prior,
    lr, mse_lambda,
    opt=torch.optim.Adam, num_its=100, construction='0', thresh=None, parallel=True):
    """Performs topological post-processing and returns refined model predictions.

    Args:
        inputs (torch.Tensor): Model's raw output (logits) with shape [1, num_classes, depth, height, width].
        model (torch.nn.Module): Pre-trained segmentation model (used only for inference).
        prior (dict): Topological prior (expected Betti numbers per class).
        lr (float): Learning rate for optimizer.
        mse_lambda (float): Weighting factor for similarity constraint.
        opt (torch.optim.Optimizer): Optimizer type (default: Adam).
        num_its (int): Number of optimization iterations.
        construction (str): '0' for 6-connectivity (3D) or 'N' for 26-connectivity (3D).
        thresh (float): Threshold for defining foreground in topological post-processing.
        parallel (bool): Use multiprocessing for homology computation.

    Returns:
        torch.Tensor: Refined segmentation output (logits).
    """
    # Get working device
    device = inputs.device

    # Get raw prediction logits
    model.eval()
    with torch.no_grad():
        pred_unet = torch.softmax(model(inputs), 1).detach().squeeze()

    # Define ROI if needed
    if thresh:
        roi = get_roi(pred_unet[1:].sum(0).squeeze(), thresh)
    else:
        roi = [slice(None, None)] + [slice(None, None) for dim in range(len(spatial_xyz))]
    
    # Initialise topological model and optimiser
    model_topo = copy.deepcopy(model)
    model_topo.eval()
    optimiser = opt(model_topo.parameters(), lr=lr)

    # Convert prior to tensor
    max_dims = [len(b) for b in prior.values()]
    prior = {torch.tensor(c, device=device): torch.tensor(b, device=device) for c, b in prior.items()}

    # Set persistent homology computation method
    PH = {'0': crip_wrapper, 'N': trip_wrapper}

    for _ in range(num_its):
        optimiser.zero_grad()

        # Compute new model prediction
        outputs = torch.softmax(model_topo(inputs), 1).squeeze()
        outputs_roi = outputs[roi]

        # Build class/combination-wise (c-wise) image tensor for prior
        combos = torch.stack([outputs_roi[c.T].sum(0) for c in prior.keys()])

        # Compute persistence barcodes
        combos_arr = combos.detach().cpu().numpy().astype(np.float64)
        if parallel:
            with torch.no_grad():
                with Pool(len(prior)) as p:
                    bcodes_arr = p.starmap(PH[construction], zip(combos_arr, max_dims))
        else:
            with torch.no_grad():
                bcodes_arr = [PH[construction](combo, max_dim) for combo, max_dim in zip(combos_arr, max_dims)]

        # Convert barcodes to differentiable form
        max_features = max(len(bc) for bc in bcodes_arr)
        bcodes = torch.zeros([len(prior), max(max_dims), max_features], device=device)
        for c, (combo, bcode) in enumerate(zip(combos, bcodes_arr)):
            _, fin = get_differentiable_barcode(combo, bcode)
            for dim in range(len(fin)):
                bcodes[c, dim, :len(fin[dim])] = fin[dim]

        # Match barcodes to prior
        stacked_prior = torch.stack(list(prior.values()))
        stacked_prior.T[0] -= 1  # Remove infinite persistence component
        matching = torch.zeros_like(bcodes).bool()
        for c, combo in enumerate(stacked_prior):
            for dim in range(len(combo)):
                matching[c, dim, :combo[dim]] = True

        # Compute persistence losses
        A = (1 - bcodes[matching]).sum()
        Z = bcodes[~matching].sum()
        mse = torch.nn.functional.mse_loss(outputs, pred_unet)

        # Get similarity constraint
        mse = F.mse_loss(outputs, pred_unet)

        # Optimisation
        loss = A + Z + mse_lambda * mse
        loss.backward()
        optimiser.step()

    return model_topo

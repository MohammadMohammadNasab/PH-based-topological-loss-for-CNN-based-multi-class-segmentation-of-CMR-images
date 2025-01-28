from multiprocessing import Pool
import cripser as crip
import tcripser as trip
import numpy as np
import torch
import torch.nn.functional as F
import cripser
import copy
from torch.optim import SGD, Adam

def crip_wrapper(X, D):
    return crip.computePH(X, maxdim=D)

def trip_wrapper(X, D):
    return trip.computePH(X, maxdim=D)

def get_roi(X, thresh=0.01):
    true_points = torch.nonzero(X >= thresh)
    corner1 = true_points.min(dim=0)[0]
    corner2 = true_points.max(dim=0)[0]
    roi = [slice(None, None)] + [slice(c1, c2 + 1) for c1, c2 in zip(corner1, corner2)]
    return roi

def get_differentiable_barcode(tensor, barcode):
    '''Makes the barcode returned by CubicalRipser differentiable using PyTorch.
    Note that the critical points of the CubicalRipser filtration reveal changes in sub-level set topology.
    
    Arguments:
        REQUIRED
        tensor  - PyTorch tensor w.r.t. which the barcode must be differentiable
        barcode - Barcode returned by using CubicalRipser to compute the PH of tensor.numpy() 
    '''
    # Identify connected component of ininite persistence (the essential feature)
    inf = barcode[barcode[:, 2] == np.finfo(barcode.dtype).max]
    fin = barcode[barcode[:, 2] < np.finfo(barcode.dtype).max]
    
    # Get birth of infinite feature
    inf_birth = tensor[tuple(inf[:, 3:3+tensor.ndim].astype(np.int64).T)]
    
    # Calculate lifetimes of finite features
    births = tensor[tuple(fin[:, 3:3+tensor.ndim].astype(np.int64).T)]
    deaths = tensor[tuple(fin[:, 6:6+tensor.ndim].astype(np.int64).T)]
    delta_p = (deaths - births)
    
    # Split finite features by dimension
    delta_p = [delta_p[fin[:, 0] == d] for d in range(tensor.ndim)]
    
    # Sort finite features by persistence
    delta_p = [torch.sort(d, descending=True)[0] for d in delta_p]
    
    return inf_birth, delta_p

def multi_class_topological_post_processing(
    inputs, model, prior,
    lr, mse_lambda,
    opt=torch.optim.Adam, num_its=100, construction='0', thresh=None, parallel=True):
    '''Performs topological post-processing.
    
    Arguments:
        REQUIRED
        inputs       - PyTorch tensor - [batch_size, number of classes] + [spatial dimensions (2D or 3D)]
        model        - Pre-trained CNN as PyTorch module (without final activation)
        prior        - Topological prior as dictionary:
                       keys are tuples specifying the channel(s) of inputs
                       values are tuples specifying the desired Betti numbers
        lr           - Learning rate for SGD optimiser
        mse_lambda   - Weighting for similarity constraint
        
        OPTIONAL [default]
        opt          - PyTorch optimiser [torch.optim.Adam]
        num_its      - Iterable of number iterations(s) to run for each scale [100]
        construction - Either '0' (4 (2D) or 6 (3D) connectivity) or 'N' (8 (2D) or 26 (3D) connectivity) ['0']
        thresh       - Threshold at which to define the foreground ROI for topological post-processing
    '''
    
    # Get image properties
    batch_size = inputs.shape[0]
    spatial_xyz = list(inputs.shape[2:])
    device = inputs.device
    
    # Get raw prediction
    model.eval()
    with torch.no_grad():
        pred_unet = torch.softmax(model(inputs), 1).detach()
    
    # Initialize single topological model and optimizer
    model_topo = copy.deepcopy(model)
    model_topo.eval()
    optimiser = opt(model_topo.parameters(), lr=lr)
    
    # Inspect prior and convert to tensor
    max_dims = [len(b) for b in prior.values()]
    prior_tensor = {torch.tensor(c): torch.tensor(b) for c, b in prior.items()}
    
    # Set mode of cubical complex construction
    PH = {'0': crip_wrapper, 'N': trip_wrapper}

    for it in range(num_its):
        # Reset gradients
        optimiser.zero_grad()
        
        # Get current predictions for all batches
        outputs = torch.softmax(model_topo(inputs), 1)
        
        # Initialize batch losses
        batch_loss_A = 0
        batch_loss_Z = 0
        batch_loss_mse = 0
        
        # Process each batch
        for batch_idx in range(batch_size):
            if thresh:
                roi = get_roi(pred_unet[batch_idx, 1:].sum(0).squeeze(), thresh)
            else:
                roi = [slice(None, None)] + [slice(None, None) for dim in range(len(spatial_xyz))]
            
            outputs_roi = outputs[batch_idx][roi]
            
            # Build class/combination-wise image tensor for prior
            combos = torch.stack([outputs_roi[c.permute(*torch.arange(c.ndim - 1, -1, -1))].sum(0) 
                                for c in prior_tensor.keys()])
            
            # Invert probabilistic fields
            combos = 1 - combos
            
            # Get barcodes
            combos_arr = combos.detach().cpu().numpy().astype(np.float64)
            if parallel:
                with torch.no_grad():
                    with Pool(len(prior)) as p:
                        bcodes_arr = p.starmap(PH[construction], zip(combos_arr, max_dims))
            else:
                with torch.no_grad():
                    bcodes_arr = [PH[construction](combo, max_dim) 
                                for combo, max_dim in zip(combos_arr, max_dims)]
            
            # Process barcodes
            max_features = max([bcode_arr.shape[0] for bcode_arr in bcodes_arr])
            bcodes = torch.zeros([len(prior), max(max_dims), max_features], 
                               requires_grad=False, device=device)
            
            for c, (combo, bcode) in enumerate(zip(combos, bcodes_arr)):
                _, fin = get_differentiable_barcode(combo, bcode)
                for dim in range(len(spatial_xyz)):
                    bcodes[c, dim, :len(fin[dim])] = fin[dim]
            
            # Select features for loss
            stacked_prior = torch.stack(list(prior_tensor.values()))
            stacked_prior.T[0] -= 1
            matching = torch.zeros_like(bcodes).detach().bool()
            for c, combo in enumerate(stacked_prior):
                for dim in range(len(combo)):
                    matching[c, dim, slice(None, stacked_prior[c, dim])] = True
            
            # Accumulate batch losses
            batch_loss_A += (1 - bcodes[matching]).sum()
            batch_loss_Z += bcodes[~matching].sum()
            batch_loss_mse += F.mse_loss(outputs[batch_idx], pred_unet[batch_idx])
        
        # Average losses over batch
        loss = (batch_loss_A + batch_loss_Z) / batch_size + mse_lambda * batch_loss_mse / batch_size
        loss.backward()
        optimiser.step()
    
    return model_topo

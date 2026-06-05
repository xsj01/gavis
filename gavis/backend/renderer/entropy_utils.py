import torch
from typing import Generator, Optional, Tuple, Union
from jaxtyping import Float, Int
from torch import Tensor, nn
import numpy as np


def format_color(color, transmittance):
    # color & color variance is curruently [3 * num samples, ]
    # needs to be [num samples, 3]
    # furthermore, need to prune gaussians that have 0 transmittance
    length = color.shape[0]
    assert length % 3 == 0

    mask = torch.empty(length, device=torch.device("cuda"))
    mask[::3] = transmittance
    mask[1::3] = transmittance
    mask[2::3] = transmittance
    color = color[mask != -1]

    length = color.shape[0]

    formatted_tensor = torch.empty(length // 3, 3, device=torch.device("cuda"))
    formatted_tensor[:,0] = color[::3]
    formatted_tensor[:,1] = color[1::3]
    formatted_tensor[:,2] = color[2::3]

    return formatted_tensor

def gmm_entropy_upper_bound_huber(
        mean : Float[Tensor, "*bs num_samples 1"],
        variance : Float[Tensor, "*bs num_samples 3"],
        weights : Float[Tensor, "*bs num_samples 1"], 
        ray_indices: Optional[Int[Tensor, "num_samples"]] = None,
        num_rays: Optional[int] = None,
                            ) -> Float[Tensor, "*bs 1"]:
    '''
    Calculate upper bound on entropy of a Gaussian mixture model using Huber's method
    https://isas.iar.kit.edu/pdf/MFI08_HuberBailey.pdf
    '''
    weights_normalized = weights
    temp = torch.zeros(num_rays, 1, device=mean.device, dtype=mean.dtype)
    temp = temp.index_add_(0, ray_indices, weights_normalized)

    if not torch.allclose(temp, torch.ones_like(temp)):
        # pass
        print('got unnormalized weights!!')
        breakpoint()

    e1 = torch.log(torch.max(variance,torch.tensor(1e-9))).sum(dim=-1, keepdim=True)
    e1 += 0.5 * 3 * np.log(2 * np.pi * np.e ) 
    entropy = weights * torch.nan_to_num( -torch.log(torch.max(weights,torch.tensor(1e-9))) + e1)

    temp = torch.zeros(num_rays, 1, device=mean.device, dtype=mean.dtype)
    entropy = temp.index_add_(0, ray_indices, entropy)
    
    return entropy

def gmm_entropy_upper_bound(mean : Float[Tensor, "*bs num_modes dim"], 
                            variance : Float[Tensor, "*bs num_modes dim"],
                            weights : Float[Tensor, "*bs num_modes 1"],
                            ray_indices: Optional[Int[Tensor, "num_samples"]] = None,
                            num_rays: Optional[int] = None,
                            ) -> Float[Tensor, "*bs 1"]:
    '''
    Multi gaussian -> single gaussian
    Calculate upper bound on entropy of a Gaussian mixture model
    '''
    # Ensure weights sum to 1
    weights_normalized = weights
    temp = torch.zeros(num_rays, 1, device=mean.device, dtype=mean.dtype)
    temp = temp.index_add_(-2, ray_indices, weights_normalized)
    if not torch.allclose(temp, torch.ones_like(temp)):
        print('got unnormalized weights!!')   

    d = variance.size(-1)
    
    # Calculate the mean of the GMM

    mean_gmm = torch.zeros(num_rays, d, device=mean.device, dtype=mean.dtype)
    # breakpoint()
    mean_gmm = mean_gmm.index_add_(-2, ray_indices, weights_normalized * mean)
    mean_gmm = mean_gmm.unsqueeze(-2)

    # Expand mean_gmm for the upcoming calculations
    mean_gmm_expanded = mean_gmm.unsqueeze(-1)
    
    # Convert the variance vectors to diagonal matrices
    variance_diag = torch.diag_embed(variance)

    mean = mean.unsqueeze(-1)
    weights_normalized = weights_normalized.unsqueeze(-1)
    
    # Calculate the outer product of the means
    mean_outer_product = torch.matmul( mean,  mean.transpose(-1, -2))

    # Calculate the variance of the GMM
    variance_gmm = torch.zeros(num_rays, 3, 3, device=mean.device, dtype=mean.dtype)
    # breakpoint()
    variance_gmm = variance_gmm.index_add_(-3, ray_indices, weights_normalized * (variance_diag + mean_outer_product))
    variance_gmm = variance_gmm.unsqueeze(-3)

    variance_gmm -= torch.matmul(mean_gmm_expanded , mean_gmm_expanded.transpose(-1, -2))
    
    # Compute the log determinant of the covariance matrix for numerical stability
    log_det_variance_gmm = torch.logdet(variance_gmm)
    

    # Calculate the entropy of the single Gaussian
    entropy = 0.5 * (log_det_variance_gmm + d * (np.log(2 * np.pi) + 1))

    return entropy

def gmm_nll(x: Float[Tensor, "*bs 1"],
            mean : Float[Tensor, "*bs num_modes dim"], 
            variance : Float[Tensor, "*bs num_modes dim"],
            weights : Float[Tensor, "*bs num_modes 1"],
            ray_indices: Optional[Int[Tensor, "num_samples"]] = None,
            num_rays: Optional[int] = None,
            ) -> Float[Tensor, "*bs 1"]:


    flatten = True if ray_indices is not None and num_rays is not None else False
    
    if flatten:
        mean = mean.unsqueeze(0)
        variance = variance.unsqueeze(0)
        weights = weights.unsqueeze(0)

    elif len(x.shape) < len(mean.shape):
        x = x.unsqueeze(-2)

    # Ensure variances are positive to avoid numerical issues
    min_variance = 1e-4
    variance = torch.clamp(variance, min=min_variance)
    
    # breakpoint()
    # Calculate the exponent term in the Gaussian formula
    if flatten:
        x_expanded = x.index_select(0, ray_indices)
        exponent_term = -0.5 * ((x_expanded - mean) ** 2) / variance
    else:
        exponent_term = -0.5 * ((x - mean) ** 2) / variance
    
    # Compute the log probability of each component
    log_prob = exponent_term - torch.log(torch.sqrt(2 * torch.pi * variance))
    
    # Sum over the dimensions
    log_prob = log_prob.sum(dim=-1) 
    
    # Combine the log probability with the weights and apply log-sum-exp for numerical stability
    if flatten:
        # breakpoint()
        exps = torch.exp(log_prob)*weights.squeeze(-1)
        # log_prob = flattened_logsumexp(log_prob.squeeze(0), ray_indices, num_rays)
        exps = exps.squeeze(0)
        exp_sums = torch.zeros(num_rays).to(exps)
        exp_sums.index_add_(0, ray_indices, exps)
        
        selector = exp_sums==0.
        # breakpoint()
        ll = torch.log(torch.clamp(exp_sums, min=1e-7))
        ll[selector]=0.

        ll = ll/(~selector).sum()*selector.shape[0]
        # breakpoint()
    else:
        # breakpoint()
        # log_prob = log_prob + torch.log(weights.squeeze(-1))
        log_prob = log_prob + torch.log(weights.squeeze(-1))
        # print('log_prob', log_prob)
        ll = torch.logsumexp(log_prob, dim=-1)
        # exp = (torch.exp(log_prob)*weights.squeeze(-1)).sum(dim=-1)
        # ll = torch.log(torch.clamp(exp, min=1e-8))

    # breakpoint()
    return -ll

def flattened_logsumexp(v, index, num_batches):
    exps = torch.exp(v)
    

    exp_sums = torch.zeros(num_batches, device=v.device)
    exp_sums.index_add_(0, index, exps)
    # breakpoint()
    

    logsumexp = torch.log(exp_sums)#.index_select(0, index)
    
    return logsumexp
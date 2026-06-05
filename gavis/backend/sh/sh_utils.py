import numpy as np
from scipy.special import lpmv, sph_harm, spherical_in, sph_harm_y
from scipy.special import factorial

import matplotlib.pyplot as plt

import torch
# from e3nn import o3
# from e3nn.o3 import wigner_D

# from .sh_utils import *

VMF_USE_BIAS = True

def real_sph_harm(ell, m, theta, phi):
    """
    Compute real spherical harmonic Y_ell^m(theta, phi).
    Uses SciPy's complex sph_harm and converts to a real basis.
    """
    # complex value
    # Yc = sph_harm(abs(m), ell, phi, theta)  # note: SciPy uses (m, l, phi, theta) convention
    Yc = sph_harm_y(ell, abs(m), theta, phi)
    # Yc = sph_harm(m, ell, phi, theta)  # note: SciPy uses (m, l, phi, theta) convention
    if m < 0:
        return np.sqrt(2) * (-1)**m * Yc.imag
        # return np.sqrt(2)* Yc.imag
    elif m == 0:
        return Yc.real
    else:
        return np.sqrt(2) * (-1)**m * Yc.real
        # return np.sqrt(2)* Yc.real

def query_sh_scipy(coeff, x):
    """
    Evaluate a batch of real spherical-harmonic expansions at directions x.
    
    Parameters
    ----------
    coeff : ndarray, shape (B, L, 2*L+1)
        SH coefficients for degrees 0..L-1 and orders -L..+L.
    x : ndarray, shape (B, 3)
        Raw Cartesian direction vectors (need not be unit length).
    
    Returns
    -------
    values : ndarray, shape (B,)
        The function values at each direction.
    """
    B, Lp1, M = coeff.shape
    L = Lp1 - 1
    assert M == 2*L + 1, "coeff third dimension must be 2*L+1"
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    if isinstance(coeff, torch.Tensor):
        coeff = coeff.cpu().numpy()
    # Normalize directions
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    dirs = x / norms
    xs, ys, zs = dirs[:,0], dirs[:,1], dirs[:,2]
    # Angles
    theta = np.arccos(np.clip(zs, -1.0, 1.0))   # colatitude :contentReference[oaicite:5]{index=5}
    phi   = np.arctan2(ys, xs)                  # azimuth

    # Accumulate result
    result = np.zeros(B, dtype=float)
    for ell in range(L+1):
        for m in range(-ell, ell+1):
            idx = m + L
            Ylm = real_sph_harm(ell, m, theta, phi)
            result += coeff[:, ell, idx] * Ylm
    return torch.from_numpy(result)

def query_sh(coeff, x):
    """
    Evaluate a batch of real spherical-harmonic expansions at directions x.
    
    Parameters
    ----------
    coeff : ndarray, shape (B, L, 2*L+1)
        SH coefficients for degrees 0..L-1 and orders -L..+L.
    x : ndarray, shape (B, 3)
        Raw Cartesian direction vectors (need not be unit length).
    
    Returns
    -------
    values : ndarray, shape (B,)
        The function values at each direction.
    """
    # B, Lp1, M = coeff.shape
    # L = Lp1 - 1
    # assert M == 2*L + 1, "coeff third dimension must be 2*L+1"
   
    # Normalize directions
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    if not isinstance(coeff, torch.Tensor):
        coeff = torch.tensor(coeff)

    dirs = x / torch.norm(x, dim=-1, keepdim=True)

    B = coeff.shape[0]

    if coeff.dim() == 2:
        # flattened coefficients
        N = int(np.sqrt(coeff.shape[1]))
        assert N*N == coeff.shape[1], "Flattened coefficients must be square"

        L = N - 1
        Ylm = rsh_fn[L](dirs)  # (B, L^2)
        result = (Ylm * coeff).sum(dim=-1)  # (B,)
    elif coeff.dim() == 3:
        
        # full coefficients
        L = coeff.shape[1] - 1
        M = coeff.shape[2]
        assert M == 2*L + 1, "coeff third dimension must be 2*L+1"

        Ylm = unflatten_sh(rsh_fn[L](dirs))  # (B, L+1, 2*L+1)
        result = (Ylm * coeff).sum(dim=(-1, -2))  # (B,)
    else:
        raise ValueError("coeff must be 2D (flattened) or 3D (unflattened SH coefficients)")
    
    return result


def von_mises2sh_north_pole(sh_degree, k):
    """
    Compute SH coefficients of exp(k x·μ - k) for μ at the north pole.
    
    Parameters
    ----------
    sh_degree : int
        Maximum spherical-harmonic degree ℓ (inclusive).
    k : float
        Concentration parameter.
    
    Returns
    -------
    coefs : ndarray, shape (1, sh_degree+1, 2*sh_degree+1)
        SH coefficients c_{ℓm}, with m offset by +sh_degree.
        Only the central m=0 entries are nonzero.
    """
    L = sh_degree
    coefs = np.zeros((1, L+1, 2*L+1))
    for ell in range(L+1):
        # Normalization for Y_{ℓ0}
        N_l0 = np.sqrt((2*ell + 1) / (4*np.pi))
        # Expansion coefficient
        coef = 4*np.pi * N_l0 * np.exp(-k) * spherical_in(ell, k)
        # Place at m=0 index (offset L)
        coefs[0, ell, L] = coef
    return coefs

vMF_coeffs = {}

def get_vMF_coeffs(sh_degree, k, device=None, dtype=torch.float32, flatten=False):
    """
    Get or compute the von Mises-Fisher coefficients for a given SH degree and concentration.
    
    Parameters
    ----------
    sh_degree : int
        Maximum spherical-harmonic degree ℓ (inclusive).
    k : float
        Concentration parameter.
    
    Returns
    -------
    coeffs : ndarray, shape (1, sh_degree+1, 2*sh_degree+1)
        SH coefficients for the von Mises-Fisher distribution.
    """
    if type(k) is torch.Tensor:
        k = k.item()
    k = round(k, 5)  # round to avoid floating-point issues
    sh_degree = int(sh_degree)
    L = sh_degree
    if (sh_degree, k) in vMF_coeffs:
        return vMF_coeffs[(sh_degree, k)].to(device=device, dtype=dtype)
    C = []

    for ell in range(L+1):
        N_l0 = np.sqrt((2*ell+1)/(4*np.pi))
        
        C.append(4*np.pi * N_l0 * np.exp(-k) * spherical_in(ell, k))
    
    C = torch.tensor(C, dtype=dtype, device=device)  # (L+1,)
    C /= torch.sqrt((2*torch.arange(L+1).to(C) + 1) / (4*torch.pi))
    
    if VMF_USE_BIAS:
        print('before bias:', C)
        zeta = 1.0 / (np.exp(k) - np.exp(-k))
        xi = -np.exp(-k) / (np.exp(k) - np.exp(-k))

        C = C * (zeta / np.exp(-k))
        C[0] += 4 * np.pi * xi
        print('after bias:', C)

    if flatten:
        C = flatten_sh(C.unsqueeze(0).unsqueeze(2).repeat(1,1,2*L+1))
    vMF_coeffs[(sh_degree, k)] = C
    return C

from .rsh import rsh_fn
def vMF2SH(sh_degree: int, k: float, mu: torch.Tensor, flatten: bool = False) -> torch.Tensor:
    """
    SH coeffs of exp(k x·mu - k) for batch of mu (N×3).
    Uses closed‐form c_{lm} = C_l * Y_lm*(mu).
    Returns: Tensor (N, L+1, 2L+1) of real coefficients.
    """
    # 1) north‐pole radial factors
    L = sh_degree
    C = get_vMF_coeffs(L, k, device=mu.device, dtype=mu.dtype, flatten=flatten)  # (L+1,)
    # C = []
    # for ell in range(L+1):
    #     N_l0 = np.sqrt((2*ell+1)/(4*np.pi))
    #     C.append(4*np.pi * N_l0 * np.exp(-k) * spherical_in(ell, k))
    # C = torch.tensor(C, dtype=mu.dtype, device=mu.device)  # (L+1,)

    # normalize mu
    mu = mu / mu.norm(dim=-1, keepdim=True)  # ensure unit
    # 2) spherical angles of each mu
    # x,y,z = mu.unbind(-1)
    # theta = torch.acos(z.clamp(-1,1))  # [0,π]
    # phi   = torch.atan2(y, x)          # [-π,π]

    N = mu.shape[0]

    if not flatten:
        # out = torch.zeros((N, L+1, 2*L+1), device=mu.device, dtype=mu.dtype)

        out = unflatten_sh(rsh_fn[L](mu)) * C.unsqueeze(0).unsqueeze(2) # (N, L+1, 2L+1)

        # norm = torch.sqrt((2*torch.arange(L+1, device=mu.device, dtype=mu.dtype) + 1) / (4*torch.pi))  # (L+1,)
        # norm = norm.unsqueeze(0).unsqueeze(2)  # (1, L+1, 1)
        # out *= C #/ norm  # (N, L+1, 2L+1)
    else:
        out = rsh_fn[L](mu) * C

        # C = C.unsqueeze(0).unsqueeze(2)  # (1, L+1, 1)
        # norm = torch.sqrt((2*torch.arange(L+1, device=mu.device, dtype=mu.dtype) + 1) / (4*torch.pi))  # (L+1,)
        # norm = norm.unsqueeze(0).unsqueeze(2)  # (1, L+1, 1)
        # C_norm = C / norm  # (1, L+1, 1)
        # out *= flatten_sh(C.repeat(1,1,2*L+1))  # (N, L+1, 2L+1)


    # # 3) fill via Y_lm(mu)
    # for ell in range(L+1):
    #     norm = np.sqrt((2*ell+1)/(4*np.pi))
    #     for m in range(-ell, ell+1):
    #         # SciPy sph_harm: sph_harm(m, ell, phi, theta)
    #         # Ylm = sph_harm(m, ell, phi.cpu().numpy(), theta.cpu().numpy())
    #         # Ylm = torch.from_numpy(Ylm).to(mu.device)  # complex128
    #         # # get real basis
    #         # if m < 0:
    #         #     R = np.sqrt(2)*(-1)**m * Ylm.imag
    #         # elif m == 0:
    #         #     R =  Ylm.real
    #         # else:
    #         #     R = np.sqrt(2)*(-1)**m * Ylm.real
    #         Ylm = real_sph_harm(ell, m, theta.cpu().numpy(), phi.cpu().numpy())
    #         R = torch.from_numpy(Ylm).to(mu.device)  # real64
    #         out[:, ell, m+L] = C[ell] * R /norm
    
    # normalize by peak value
    # pole_vals = query_sh(out.cpu().numpy(), mu.cpu().numpy())       # shape (N,)
    # out = out / pole_vals[:, None, None]
    return out#.to(mu)

def von_mises2sh_scipy(sh_degree: int, k: float, mu: torch.Tensor):
    """
    SH coeffs of exp(k x·mu - k) for batch of mu (N×3).
    Uses closed‐form c_{lm} = C_l * Y_lm*(mu).
    Returns: Tensor (N, L+1, 2L+1) of real coefficients.
    """
    # 1) north‐pole radial factors
    L = sh_degree
    C = []
    for ell in range(L+1):
        N_l0 = np.sqrt((2*ell+1)/(4*np.pi))
        C.append(4*np.pi * N_l0 * np.exp(-k) * spherical_in(ell, k))
    C = torch.tensor(C, dtype=mu.dtype, device=mu.device)  # (L+1,)

    # normalize mu
    mu = mu / mu.norm(dim=-1, keepdim=True)  # ensure unit
    # 2) spherical angles of each mu
    x,y,z = mu.unbind(-1)
    theta = torch.acos(z.clamp(-1,1))  # [0,π]
    phi   = torch.atan2(y, x)          # [-π,π]

    N = mu.shape[0]
    out = torch.zeros((N, L+1, 2*L+1), device=mu.device, dtype=mu.dtype)

    # 3) fill via Y_lm(mu)
    for ell in range(L+1):
        norm = np.sqrt((2*ell+1)/(4*np.pi))
        for m in range(-ell, ell+1):
            # SciPy sph_harm: sph_harm(m, ell, phi, theta)
            # Ylm = sph_harm(m, ell, phi.cpu().numpy(), theta.cpu().numpy())
            # Ylm = torch.from_numpy(Ylm).to(mu.device)  # complex128
            # # get real basis
            # if m < 0:
            #     R = np.sqrt(2)*(-1)**m * Ylm.imag
            # elif m == 0:
            #     R =  Ylm.real
            # else:
            #     R = np.sqrt(2)*(-1)**m * Ylm.real
            Ylm = real_sph_harm(ell, m, theta.cpu().numpy(), phi.cpu().numpy())
            R = torch.from_numpy(Ylm).to(mu.device)  # real64
            out[:, ell, m+L] = C[ell] * R /norm
    
    # normalize by peak value
    # pole_vals = query_sh(out.cpu().numpy(), mu.cpu().numpy())       # shape (N,)
    # out = out / pole_vals[:, None, None]
    
    return out.to(mu)

from .sh2f import sh2f_channel
from .f2sh import f2sh_channel
from .fft import FFT_channel, FFT_channel2



def sh2f2sh(sh_coeff: torch.Tensor) -> torch.Tensor:
    """
    Convert SH coefficients to Fourier coefficients.
    
    Args:
        sh_coeff: Tensor of shape (B, L+1, 2L+1) for SH coefficients.
    
    Returns:
        Tensor of shape (B, L+1, 2L+1) representing Fourier coefficients.
    """
    B, Lp1, M = sh_coeff.shape
    assert M == 2 * (Lp1 - 1) + 1, "Invalid shape for SH coefficients"
    
    sh2f_bases = sh2f_bases_dict[Lp1].to(sh_coeff.device)
    f2sh_bases = f2sh_bases_dict[Lp1].to(sh_coeff.device)
    f = sh2f_channel(sh_coeff, sh2f_bases).contiguous()
    f1 = torch.fft.fft2(f, norm="forward").contiguous()  # Fourier transform
    f2 = torch.fft.ifft2(f1, norm="forward")  # Inverse Fourier transform to get real coefficients
    # torch.allclose(f2.real.to(f), f, atol=1e-6)
    # print((f2.real-f).norm())
    # breakpoint()
    f = f2  # Ensure real output
    sh = f2sh_channel(f, f2sh_bases)
    return sh.real


def sh_product(sh_a: torch.Tensor, sh_b: torch.Tensor, fft_return_real=True, bases=None) -> torch.Tensor:
    """
    Compute the product of two SH expansions.
    
    Args:
        sh_a: Tensor of shape (B, L+1, 2L+1) for first SH coefficients.
        sh_b: Tensor of shape (B, L+1, 2L+1) for second SH coefficients.
    
    Returns:
        Tensor of shape (B, L+1, 2L+1) representing the product.
    """
    B, Lp1, M = sh_a.shape
    assert sh_b.shape == (B, Lp1, M), "Shapes of input SH coefficients must match"
    
    in1 = sh_a
    in2 = sh_b

    L = Lp1-1
    if bases is None:
        sh2f_bases, f2sh_bases = sh2f_bases_dict[Lp1], f2sh_bases_dict[Lp1]
        sh2f_bases, f2sh_bases = sh2f_bases.to(in1), f2sh_bases.to(in2)
    else:
        sh2f_bases, f2sh_bases = bases
        # sh2f_bases, f2sh_bases = sh2f_bases.to(in1), f2sh_bases.to(in2)
    
    in1_fourier, in2_fourier = sh2f_channel(in1, sh2f_bases), sh2f_channel(in2, sh2f_bases)
    # breakpoint()
    
    out_fourier = FFT_channel2(in1_fourier, in2_fourier, return_real=fft_return_real)

    # currently only able to run on CPU due to cuFFT error: CUFFT_INTERNAL_ERROR
    # out_fourier = FFT_channel2(in1_fourier.cpu(), in2_fourier.cpu(), return_real=fft_return_real)
    # out_fourier = out_fourier.to(in1.device)  # move back to original device

    # high frequency filtering
    # start = Lp1 
    # end = start + Lp1*2 -1
    # out_fourier = out_fourier[:, start:end, start:end]
    # plt.imshow(out_fourier[0].cpu().numpy(), cmap='viridis')
    # plt.pause(0.1)
    # breakpoint()
    # breakpoint()

    res = f2sh_channel(out_fourier, f2sh_bases)
    # if not fft_return_real:
    #     res = res.real
    return res

def sh_product_orig(sh_a: torch.Tensor, sh_b: torch.Tensor, fft_return_real=True) -> torch.Tensor:
    """
    Compute the product of two SH expansions.
    
    Args:
        sh_a: Tensor of shape (B, L+1, 2L+1) for first SH coefficients.
        sh_b: Tensor of shape (B, L+1, 2L+1) for second SH coefficients.
    
    Returns:
        Tensor of shape (B, L+1, 2L+1) representing the product.
    """
    B, Lp1, M = sh_a.shape
    assert sh_b.shape == (B, Lp1, M), "Shapes of input SH coefficients must match"
    
    in1 = sh_a
    in2 = sh_b

    L = Lp1-1

    sh2f_bases, f2sh_bases = sh2f_bases_dict[Lp1], f2sh_bases_dict[2 * Lp1 - 1]
    sh2f_bases, f2sh_bases = sh2f_bases.to(in1), f2sh_bases.to(in2)

    in1_fourier, in2_fourier = sh2f_channel(in1, sh2f_bases), sh2f_channel(in2, sh2f_bases)
    # breakpoint()
    out_fourier = FFT_channel(in1_fourier, in2_fourier, return_real=fft_return_real)

    res = f2sh_channel(out_fourier, f2sh_bases)
    # if not fft_return_real:
    #     res = res.real
    return res




def update_visibility(coeff,k, mu, transmittance=1.0):
    """
    Update visibility based on SH coefficients and direction vectors.
    
    Args:
        coeff: Tensor of shape (B, L+1, 2L+1) for SH coefficients.
        mu: Tensor of shape (B, 3) for direction vectors.
        k: Tensor of shape (B,) for concentration parameters.
    
    Returns:
        Updated visibility tensor of shape (B, L+1, 2L+1).
    """
    # Compute SH coefficients for the given mu and k
    vMF = vMF2SH(coeff.shape[1] - 1, k, mu)
    vMF *= transmittance

    if coeff is None:
        return vMF
    
    # Update visibility by multiplying with the SH coefficients
    updated_coeff = coeff + vMF - sh_product(coeff, vMF)
    
    return updated_coeff

def visualize_sh_azimuthal(sh_coeff, resolution=400, cmap='viridis'):
    """
    Visualize SH function on a disk: north pole at center, 
    using an azimuthal equidistant projection.
    
    Parameters
    ----------
    sh_coeff : ndarray, shape (B, L, 2*L+1)
        Batch of SH coefficients. We assume B=1 or pick the first entry.
    resolution : int
        Number of angular samples along each axis.
    cmap : str
        Matplotlib colormap name.
    """
    # pick the first batch if necessary
    coeff = sh_coeff[0:1]
    
    # build a grid in (theta, phi)
    thetas = np.linspace(0, np.pi, resolution)     # colatitude from pole (0) to opposite (π)
    phis   = np.linspace(0, 2*np.pi, resolution)   # azimuth full circle
    Θ, Φ = np.meshgrid(thetas, phis, indexing='xy')
    
    # flatten and form batch directions
    xs = np.sin(Θ) * np.cos(Φ)
    ys = np.sin(Θ) * np.sin(Φ)
    zs = np.cos(Θ)
    dirs = np.stack([xs, ys, zs], axis=-1).reshape(-1, 3)  # (res^2, 3)
    
    # evaluate SH
    if coeff.shape[0] != dirs.shape[0]:
        # expand to match batch size
        coeff = np.repeat(coeff, dirs.shape[0], axis=0)
    vals = query_sh(coeff, dirs).reshape(resolution, resolution)
    
    # plot in polar coordinates
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    # pcolormesh expects phi axis then r axis
    mesh = ax.pcolormesh(Φ, Θ, vals, cmap=cmap, shading='auto')
    ax.set_ylim(0, np.pi)
    ax.set_theta_zero_location('N')    # zero azimuth (phi=0) at top
    ax.set_theta_direction(-1)         # increase phi clockwise
    ax.set_yticks([0, np.pi/2, np.pi])
    ax.set_yticklabels(['North\nPole', 'Equator', 'South\nPole'])
    fig.colorbar(mesh, ax=ax, label='SH value')
    plt.pause(0.1)


# 2) Plot on a 3D sphere surface
def visualize_sh_sphere(sh_coeff, resolution=100, cmap='coolwarm'):
    """
    Visualize SH function directly on a 3D sphere.
    
    Parameters
    ----------
    sh_coeff : ndarray, shape (B, L, 2*L+1)
        Batch of SH coefficients. We assume B=1 or pick the first entry.
    resolution : int
        Number of samples along theta and phi.
    cmap : str
        Matplotlib colormap name.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    
    coeff = sh_coeff[0:1]
    
    # create sphere parameterization
    thetas = np.linspace(0, np.pi, resolution)
    phis   = np.linspace(0, 2*np.pi, resolution)
    Θ, Φ = np.meshgrid(thetas, phis, indexing='xy')
    xs = np.sin(Θ) * np.cos(Φ)
    ys = np.sin(Θ) * np.sin(Φ)
    zs = np.cos(Θ)
    
    # flatten, evaluate, and reshape
    dirs = np.stack([xs, ys, zs], axis=-1).reshape(-1, 3)

    if coeff.shape[0] != dirs.shape[0]:
        # expand to match batch size
        coeff = np.repeat(coeff, dirs.shape[0], axis=0)
    vals = query_sh(coeff, dirs).reshape(resolution, resolution)
    print(vals.max(), vals.min())
    
    # normalize for coloring
    norm = plt.Normalize(vmin=vals.min(), vmax=vals.max())
    colors = plt.cm.get_cmap(cmap)(norm(vals))
    
    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(
        xs, ys, zs,
        facecolors=colors,
        rcount=resolution, ccount=resolution,
        antialiased=False, shade=False
    )
    m = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    m.set_array([])
    fig.colorbar(m, ax=ax, shrink=0.5, label='SH value')
    
    # prettify
    ax.set_box_aspect((1,1,1))
    ax.set_axis_off()
    plt.pause(0.1)


def sh_1(L):
    C = torch.zeros((L+1, 2*L + 1), dtype=float)
    # index of m=0 is offset = L-1
    # C[0, L-1] = np.sqrt(4 * np.pi)
    C[0, L] = np.sqrt(4 * np.pi)
    return C

def sh_1_minus(coeff):
    L = coeff.shape[1] - 1
    return sh_1(L) - coeff

def expand_sh(sh_coeff, L):
    """
    Expand SH coefficients to a larger degree L.
    
    Args:
        sh_coeff: Tensor of shape (B, Lp1, 2Lp1-1) for current SH coefficients.
        L: New maximum degree for expansion.
    
    Returns:
        Expanded SH coefficients of shape (B, L+1, 2L+1).
    """
    B, Lp1, M = sh_coeff.shape
    assert M == 2 * (Lp1 - 1) + 1, "Invalid shape for SH coefficients"

    assert L >= Lp1 - 1, "New degree L must be greater than or equal to current degree"
    new_coeff = torch.zeros((B, L+1, 2 * L + 1), dtype=sh_coeff.dtype, device=sh_coeff.device)
    new_coeff[:, :Lp1, L - (Lp1 - 1):L + (Lp1 - 1) + 1] = sh_coeff  # copy existing coefficients
    
    return new_coeff

def truncate_sh(sh_coeff, L):
    """
    Truncate SH coefficients to a smaller degree L.
    
    Args:
        sh_coeff: Tensor of shape (B, Lp1, 2Lp1-1) for current SH coefficients.
        L: New maximum degree for truncation.
    
    Returns:
        Truncated SH coefficients of shape (B, L+1, 2L+1).
    """
    B, Lp1, M_old = sh_coeff.shape
    L_old = Lp1 - 1
    assert M_old == 2 * L_old + 1, "Invalid shape for SH coefficients"

    assert L < L_old, "New degree L must be less than current degree"
    # new_coeff = torch.zeros((B, L+1, 2 * L + 1), dtype=sh_coeff.dtype, device=sh_coeff.device)
    new_coeff = sh_coeff[:, :L + 1, L_old - L:L_old + L + 1].contiguous()  # slice to keep only up to L
    
    return new_coeff

def flatten_sh(sh_coeff: torch.Tensor) -> torch.Tensor:
    """
    Flatten SH coefficients from shape (B, N, 2N-1) to (B, N^2),
    removing the unused entries for |m|>ℓ.

    Args:
        sh_coeff: Tensor of shape (B, N, 2N-1).

    Returns:
        Tensor of shape (B, N^2) with only the valid SH coefficients.
    """
    B, N, M = sh_coeff.shape
    assert M == 2 * N - 1, f"Expected last dim = 2N-1, got {M}"

    # 1. Build a boolean mask of shape (N, 2N-1):
    #    mask[i, j] = True  iff  |j - (N-1)| <= i
    idxs = torch.arange(M, device=sh_coeff.device)
    degrees = torch.arange(N, device=sh_coeff.device)[:, None]
    mask = (idxs[None, :] >= (N - 1 - degrees)) & (idxs[None, :] <= (N - 1 + degrees))
    # mask: (N, M) where each row i keeps exactly 2i+1 entries

    # 2. Flatten the last two dims of sh_coeff to (B, N*M)
    flat = sh_coeff.reshape(B, -1)

    # 3. Use the flattened mask to select the N^2 entries per batch
    mask_flat = mask.reshape(-1)
    out = flat[:, mask_flat]

    return out

import torch

def unflatten_sh(flat: torch.Tensor) -> torch.Tensor:
    """
    Inverse of `flatten_sh` for SH coefficients.

    Args:
        flat: Tensor of shape (B, N^2), containing only the valid SH coeffs
              (in the order defined by flatten_sh).

    Returns:
        Tensor of shape (B, N, 2N-1) where invalid entries are zero.
    """
    B, N2 = flat.shape
    N = int(N2**0.5)
    assert N * N == N2, f"Expected second dim to be a perfect square, got {N2}"
    M = 2 * N - 1

    # 1. Reconstruct the same mask we used for flattening:
    idxs = torch.arange(M, device=flat.device)
    degrees = torch.arange(N, device=flat.device)[:, None]
    mask = (idxs[None, :] >= (N - 1 - degrees)) & (idxs[None, :] <= (N - 1 + degrees))
    # mask shape: (N, M), with exactly N^2 True entries

    # 2. Prepare the output full-sized tensor (zeros by default)
    out = torch.zeros(B, N, M, device=flat.device, dtype=flat.dtype)

    # 3. Scatter the flat values back into the True positions
    #    First view `out` as (B, N*M) so we can index it with a flat mask
    out_flat = out.view(B, -1)
    mask_flat = mask.view(-1)          # length N*M, same order as flatten_sh
    out_flat[:, mask_flat] = flat      # broadcast assignment across batch

    return out


if __name__ == "__main__":
    import os
    path = os.path.dirname(os.path.abspath(__file__))
    sh2f_bases_dict = torch.load(os.path.join(path, "constants/coefficient_sh2f.pt"))
    f2sh_bases_dict = torch.load(os.path.join(path, "constants/coefficient_f2sh.pt"))
    def test_vMF():
        sh_degree = 4
        # coeff = von_mises2sh_north_pole(sh_degree, 5.)
        coeff = vMF2SH(sh_degree, 1.0, torch.tensor([[1., 0.2, 1.]]))  # North pole
        coeff = coeff.cpu().numpy()
        # breakpoint()
        visualize_sh_azimuthal(coeff, resolution=100, cmap='viridis')
        visualize_sh_sphere(coeff, resolution=100, cmap='coolwarm')
        # visualize_sh_azimuthal(np.random.rand(1, 3, 7), resolution=100, cmap='viridis')
        # visualize_sh_sphere(np.random.rand(1, 3, 7), resolution=100, cmap='coolwarm')
        breakpoint()
    
    def test_product():
        sh_degree = 15
        coeff_a = vMF2SH(sh_degree, 1.0, torch.tensor([[1., 0.2, 1.]]))
        coeff_b = vMF2SH(sh_degree, 1.0, torch.tensor([[-0.5, -0.5, -1.]]))

        coeff_1 = sh_1(sh_degree).to(coeff_a).unsqueeze(0).expand(coeff_a.shape[0], -1, -1)

        # coeff_ab = sh_product(coeff_1 - coeff_a, coeff_1 - coeff_b)
        cpeff_ab_orig = sh_product_orig(coeff_1 - coeff_a, coeff_1 - coeff_b)

        # coeff_out = coeff_1 - coeff_a - coeff_b + truncate_sh(coeff_ab, sh_degree)
        # coeff_out =  coeff_a + coeff_b - coeff_ab
        # coeff_out = coeff_1 - coeff_ab

        coeff_out = update_visibility(coeff_a, 1.0, torch.tensor([[-0.5, -0.5, -1.]]))

        coeff_out = update_visibility(coeff_out, 1.0, torch.tensor([[0.5, -0.5, -1.]]))


        # new_deg = coeff_ab.shape[1] -1
        # coeff_out_orig = expand_sh( coeff_a + coeff_b, cpeff_ab_orig.shape[1] -1) -  cpeff_ab_orig
        coeff_out_orig = expand_sh(coeff_1, cpeff_ab_orig.shape[1] - 1) - cpeff_ab_orig

        # coeff_out = sh_1(new_deg) - coeff_out

        coeff_out = coeff_out.cpu().numpy()
        coeff_out_orig = coeff_out_orig.cpu().numpy()
        visualize_sh_azimuthal(coeff_out_orig, resolution=100, cmap='viridis')
        visualize_sh_sphere(coeff_out_orig, resolution=100, cmap='coolwarm')

        visualize_sh_azimuthal(coeff_out, resolution=100, cmap='viridis')
        visualize_sh_sphere(coeff_out, resolution=100, cmap='coolwarm')
        breakpoint()
    
    def test_f2sh():
        sh_degree = 4
        coeff = vMF2SH(sh_degree, 1.0, torch.tensor([[1., 1.3, 1.]]))

        coeff2 = sh_1(sh_degree) - sh2f2sh(sh_1(sh_degree) -coeff)

        # coeff2 = sh2f2sh(coeff)

        print((coeff - coeff2).norm())
        breakpoint()

    def test_compare_gt():
        import math
        sh_degree = 4
        v1 = torch.tensor([[1., 0.2, 1.]])
        v2 = torch.tensor([[-0.5, -0.5, -1.]])
        # v2 = torch.tensor([[1., 0.2, 1.]])

        k1 = k2 = 1.0
        a1 = a2 = 1.0
        coeff1 = a1*vMF2SH(sh_degree, k1, v1)
        coeff2 = a2*vMF2SH(sh_degree, k2, v2)

        v12 = (v1*k1 + v2*k2)
        k12 = v12.norm(dim=-1, keepdim=True)
        v12 = v12 / k12  # normalize
        
        a12 = torch.exp(k12)/math.exp(k1 + k2)

        coeff12 = a12*vMF2SH(sh_degree, k12, v12)

        # coeff_result = sh_product_orig(coeff1, coeff2, fft_return_real=True)
        # coeff1_0 = coeff1[0,0,sh_degree].item()
        # coeff1[0,0,sh_degree] = 0.0
        # coeff2_0 = coeff2[0,0,sh_degree].item()
        # coeff2[0,0,sh_degree] = 0.0
        coeff_result = sh_product_orig(coeff1, coeff2, fft_return_real=True) 
        # coeff_result = sh_product(coeff1, coeff2, fft_return_real=True)

        # print((coeff - coeff_gt).norm())
        # print((coeff2 - coeff_gt2).norm())
        # print((coeff_result - coeff12).norm())
        

        visualize_sh_azimuthal(coeff_result.real.cpu().numpy(), resolution=100, cmap='viridis')
        visualize_sh_sphere(coeff_result.real.cpu().numpy(), resolution=100, cmap='coolwarm')

        visualize_sh_azimuthal(coeff12.cpu().numpy(), resolution=100, cmap='viridis')
        visualize_sh_sphere(coeff12.cpu().numpy(), resolution=100, cmap='coolwarm')
        breakpoint()

    # test_product()
    # test_f2sh()
    # breakpoint()
    test_compare_gt()
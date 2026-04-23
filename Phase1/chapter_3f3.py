# -*- coding: utf-8 -*-

import math, os, json, time
from dataclasses import dataclass, replace
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.colors as mcolors
from typing import Tuple, Dict, Any

# -----------------------------
# 0) Global config & utilities
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] {DEVICE}")

# Reproducibility
SEED = 1337
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

@dataclass
class DesignConfig:
    """Dataclass to hold all experimental parameters."""
    num_layers: int = 100           # Number of core layers
    n_min: float = 1.6              # Min refractive index (n)
    n_max: float = 2.4              # Max refractive index (n)
    d_min_nm: float = 20.0          # Min thickness (d) in nm
    d_max_nm: float = 120.0         # Max thickness (d) in nm
    air_n: float = 1.0              # Superstrate (air)
    sub_n: float = 1.5              # Substrate
    wl_min_nm: float = 350.0        # Wavelength range start (nm)
    wl_max_nm: float = 750.0        # Wavelength range end (nm)
    wl_points: int = 400            # Number of wavelength points

    # Error / grading
    grade_nm: float = 8.0           # Graded-interface width (nm); 0 => abrupt
    grade_step_nm: float = 4.0      # Sublayer size for grading (nm)
    
    # Systematic drifts
    sys_mode_n: str = "none"        # ['linear','poly','sine','exp', 'none']
    sys_mode_d: str = "none"        # independent mode for d
    sys_params_n: Dict = None       # e.g., {"a": 0.20, "p": 1.0, ...}
    sys_params_d: Dict = None

    # Random noise (Stage-2 training & MC eval)
    rnd_n_mu: float = 0.0           # Mean relative shift on n
    rnd_n_gamma: float = 0.0        # Uniform half-range on n (±gamma)
    rnd_d_mu_nm: float = 0.0        # Mean absolute shift on d (nm)
    rnd_d_gamma_nm: float = 0.0     # Uniform half-range on d (±gamma, nm)

    # Training
    lr: float = 1e-3
    # *** For real results, use 800-1000+ ***
    iters_stage1: int = 1000         # Ideal optimization
    iters_stage2: int = 1000        # Fabrication-in-loop


    # Added a fine-tuning stage
    iters_finetune: int = 300       # Extra steps at a lower learning rate
    lr_finetune: float = 1e-4       # Learning rate for polishing


    # Monte-Carlo eval
    mc_trials: int = 1000

# Wavelength grid (meters)
WAVELENGTHS_NM = torch.linspace(400.0, 700.0, 300, device=DEVICE)
WAVELENGTHS_M = WAVELENGTHS_NM * 1e-9

# Global MSE Loss
mse = nn.MSELoss()

# Output directory
OUTPUT_DIR = "ch3_figures_output_tuned_corrected"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# 1) Differentiable TMM 
# -----------------------------
def tmm_normal(n_layers, d_layers_m, wavelengths_m):
    """
    Vectorized TMM for normal incidence.
    n_layers: (L_total,) complex or real (float->cast complex)
    d_layers_m: (L_total,) meters
    wavelengths_m: (W,) meters
    Returns transmittance T of shape (W,)
    NOTE: n_layers must include air & substrate boundaries at ends.
    """
    W = wavelengths_m.shape[0]
    L = n_layers.shape[0]
    n_c = n_layers.unsqueeze(0).expand(W, -1).cfloat()
    d_c = d_layers_m.unsqueeze(0).expand(W, -1).cfloat()
    wl = wavelengths_m.unsqueeze(1).cfloat()

    k = 2 * math.pi * n_c / wl
    phi = k * d_c

    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)

    M = torch.stack([
        torch.stack([cos_phi, (-1j / n_c) * sin_phi], dim=-1),
        torch.stack([-1j * n_c * sin_phi, cos_phi], dim=-1)
    ], dim=-2)  # (W, L, 2, 2)

    Mtot = torch.eye(2, dtype=torch.cfloat, device=DEVICE).unsqueeze(0).repeat(W, 1, 1)
    # Iterate from layer 1 up to L-2 (L-1 is substrate, 0 is air)
    for i in range(1, L - 1):
        Mtot = Mtot @ M[:, i, :, :]

    n0 = n_c[:, 0]
    ns = n_c[:, -1]
    m00, m01 = Mtot[:, 0, 0], Mtot[:, 0, 1]
    m10, m11 = Mtot[:, 1, 0], Mtot[:, 1, 1]

    t = (2 * n0) / (m00*ns + m01*(ns*ns) + m10 + m11*ns) # Corrected from source
    t = (2 * n0) / (n0*(m00 + m01*ns) + (m10 + m11*ns)) # Fresnel TMM
    
    # Simpler TMM formulation
    # M_total = M_1 @ M_2 @ ... @ M_N
    # t = 2*n_0 / ( (M_00 + M_01*n_s)*n_0 + (M_10 + M_11*n_s) )
    t = (2 * n0) / ( (m00 + m01*ns)*n0 + (m10 + m11*ns) )

    T = torch.abs(t) ** 2 * (torch.real(ns) / torch.real(n0))
    return T

# -----------------------------
# 2) Graded interfaces builder 
# -----------------------------
def build_graded_stack(n_core, d_core_nm, cfg: DesignConfig):
    """
    Differentiable graded-interface builder.
    Includes explicit 0-thickness boundary layers (air/substrate).
    Returns:
        n_full: (L_total,) tensor
        d_full_m: (L_total,) meters
    """
    air = torch.as_tensor(cfg.air_n, device=n_core.device, dtype=n_core.dtype)
    sub = torch.as_tensor(cfg.sub_n, device=n_core.device, dtype=n_core.dtype)

    steps = int(math.ceil(float(cfg.grade_nm) / float(cfg.grade_step_nm))) if cfg.grade_nm > 0.0 else 0

    if steps > 0:
        s_fracs = torch.arange(1, steps + 1, device=n_core.device, dtype=n_core.dtype) / (steps + 1)
        grade_th_nm = torch.full((steps,), cfg.grade_nm / steps, device=d_core_nm.device, dtype=d_core_nm.dtype)
    else:
        s_fracs = torch.empty(0, device=n_core.device, dtype=n_core.dtype)
        grade_th_nm = torch.empty(0, device=d_core_nm.device, dtype=d_core_nm.dtype)

    n_chunks = []
    d_chunks_nm = []

    n_chunks.append(air.view(1))
    d_chunks_nm.append(torch.zeros(1, device=d_core_nm.device, dtype=d_core_nm.dtype))

    if steps > 0:
        nL = air.view(1, 1)
        nR = n_core[0].view(1, 1)
        n_g = nL + (nR - nL) * s_fracs.view(-1, 1)
        n_chunks.append(n_g.view(-1))
        d_chunks_nm.append(grade_th_nm.clone())

    L = n_core.shape[0]
    for i in range(L):
        n_chunks.append(n_core[i].view(1))
        d_chunks_nm.append(d_core_nm[i].view(1))

        if steps > 0:
            nL = n_core[i].view(1, 1)
            nR = (sub if i == L - 1 else n_core[i + 1]).view(1, 1)
            n_g = nL + (nR - nL) * s_fracs.view(-1, 1)
            n_chunks.append(n_g.view(-1))
            d_chunks_nm.append(grade_th_nm.clone())

    n_chunks.append(sub.view(1))
    d_chunks_nm.append(torch.zeros(1, device=d_core_nm.device, dtype=d_core_nm.dtype))

    n_full = torch.cat(n_chunks, dim=0)
    d_full_nm = torch.cat(d_chunks_nm, dim=0)
    d_full_m = torch.clamp(d_full_nm * 1e-9, min=1e-12)

    return n_full, d_full_m

# -----------------------------
# 3) Systematic drifts & random noise 
# -----------------------------
def make_systematic(vec, mode, params):
    """Apply deterministic depth-dependent drift to a 1D vector (n or d)."""
    if mode is None or mode == "none" or params is None:
        return vec
    z = torch.linspace(0.0, 1.0, vec.numel(), device=vec.device)
    a = float(params.get("a", 0.0))
    p = float(params.get("p", 1.0))
    f = float(params.get("f", 1.0))
    phi = float(params.get("phi", 0.0))
    b = float(params.get("b", 1.0))
    if mode == "linear":
        drift = a * z
    elif mode == "poly":
        drift = a * (z ** p)
    elif mode == "sine":
        drift = a * torch.sin(2 * math.pi * f * z + phi)
    elif mode == "exp":
        num = torch.exp(b * z) - 1.0
        den = math.exp(b) - 1.0
        drift = a * (num / (den + 1e-9))
    else:
        drift = torch.zeros_like(vec)
    return vec + drift

def add_random_uniform(vec, mu, gamma):
    """Add U(mu-gamma, mu+gamma) elementwise."""
    if gamma == 0.0 and mu == 0.0:
        return vec
    noise = (torch.rand_like(vec) * 2.0 - 1.0) * gamma + mu
    return vec + noise

# -----------------------------
# 4) Model: two-head FCNN 
# -----------------------------
class OnlineOptimizer(nn.Module):
    def __init__(self, seed_size, num_layers, n_min, n_max, d_min_nm, d_max_nm):
        super().__init__()
        hidden = 256
        self.backbone = nn.Sequential(
            nn.Linear(seed_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.out_n = nn.Sequential(nn.Linear(hidden, num_layers), nn.Sigmoid())
        self.out_d = nn.Sequential(nn.Linear(hidden, num_layers), nn.Sigmoid())
        self.n_min, self.n_max = n_min, n_max
        self.d_min, self.d_max = d_min_nm, d_max_nm

    def forward(self, seed):
        h = self.backbone(seed)
        n_hat = self.out_n(h).squeeze() * (self.n_max - self.n_min) + self.n_min
        d_hat_nm = self.out_d(h).squeeze() * (self.d_max - self.d_min) + self.d_min
        return n_hat, d_hat_nm

# -----------------------------
# 5) Core Physics/Evaluation Forward Pass
# -----------------------------
def forward_spectrum(n_core, d_core_nm, cfg: DesignConfig, with_grading=True,
                     with_systematic=True, with_random=False):
    """
    Builds the full stack and computes T(λ), applying errors as specified.
    - with_grading: Applies build_graded_stack
    - with_systematic: Applies deterministic drifts (sys_mode, sys_params)
    - with_random: Applies one sample of stochastic noise (rnd_n_mu, etc.)
    """
    
    n_eff, d_eff_nm = n_core, d_core_nm
    
    if with_systematic:
        n_eff = make_systematic(n_eff, cfg.sys_mode_n, cfg.sys_params_n)
        d_eff_nm = make_systematic(d_eff_nm, cfg.sys_mode_d, cfg.sys_params_d)

    if with_random:
        # Note: Chapter 3 applies random error as a *relative* percentage
        # n_imp = n_id + n_id * n_rand
        # Here we add relative noise for n, absolute for d
        n_rel_noise = add_random_uniform(torch.zeros_like(n_eff), cfg.rnd_n_mu, cfg.rnd_n_gamma)
        n_eff = n_eff * (1.0 + n_rel_noise)
        
        d_abs_noise_nm = add_random_uniform(torch.zeros_like(d_eff_nm), cfg.rnd_d_mu_nm, cfg.rnd_d_gamma_nm)
        d_eff_nm = d_eff_nm + d_abs_noise_nm

    # Clamp values to physical bounds after all errors
    n_eff = torch.clamp(n_eff, cfg.n_min, cfg.n_max)
    d_eff_nm = torch.clamp(d_eff_nm, cfg.d_min_nm, cfg.d_max_nm)

    if with_grading and cfg.grade_nm > 0.0:
        n_full, d_full_m = build_graded_stack(n_eff, d_eff_nm, cfg)
    else:
        # No grading; just the core stack between air and substrate
        n_full = torch.cat([
            torch.tensor([cfg.air_n], device=DEVICE, dtype=n_eff.dtype),
            n_eff,
            torch.tensor([cfg.sub_n], device=DEVICE, dtype=n_eff.dtype)
        ])
        d_full_m = torch.cat([
            torch.tensor([0.0], device=DEVICE, dtype=d_eff_nm.dtype),
            d_eff_nm * 1e-9,
            torch.tensor([0.0], device=DEVICE, dtype=d_eff_nm.dtype)
        ])
        d_full_m = torch.clamp(d_full_m, min=1e-12)


    return tmm_normal(n_full, d_full_m, WAVELENGTHS_M)

# -----------------------------
# 6) Refactored Training & Evaluation Functions with Fine Tuning
# -----------------------------

def run_optimization(cfg: DesignConfig, target_spectrum: torch.Tensor, verbose=True) -> Dict[str, Any]:
    """
    Performs the full two-stage optimization, now with fine-tuning.
    Returns a dictionary with the resulting stacks and spectra.
    """
    
    # Ensure target is on the correct device
    target_spectrum = target_spectrum.to(DEVICE)
    
    # Use a unique seed vector for this specific run
    seed_vec = torch.randn(cfg.num_layers, device=DEVICE)
    
    model = OnlineOptimizer(
        seed_size=cfg.num_layers, num_layers=cfg.num_layers,
        n_min=cfg.n_min, n_max=cfg.n_max, d_min_nm=cfg.d_min_nm, d_max_nm=cfg.d_max_nm
    ).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=cfg.lr)

    # --- Stage 1: Ideal Optimization ---
    if verbose:
        print(f"  [Opt] Running Stage 1 (Ideal) for {cfg.iters_stage1} iterations...")
    for it in range(1, cfg.iters_stage1 + 1):
        opt.zero_grad()
        n_hat, d_hat_nm = model(seed_vec)
        
        T = forward_spectrum(n_hat, d_hat_nm, cfg, 
                             with_grading=False, 
                             with_systematic=False, 
                             with_random=False)
        
        loss = mse(T, target_spectrum)
        loss.backward()
        opt.step()

    # --- !! Stage 1 Fine-Tuning !! ---
    if verbose:
        print(f"  Fine-tuning Stage 1 for {cfg.iters_finetune} iterations...")
    # Create a new optimizer with a smaller learning rate
    opt_finetune_s1 = optim.Adam(model.parameters(), lr=cfg.lr_finetune)
    for it in range(1, cfg.iters_finetune + 1):
        opt_finetune_s1.zero_grad()
        n_hat, d_hat_nm = model(seed_vec)
        
        T = forward_spectrum(n_hat, d_hat_nm, cfg, 
                             with_grading=False, 
                             with_systematic=False, 
                             with_random=False)
        
        loss = mse(T, target_spectrum)
        loss.backward()
        opt_finetune_s1.step()

    # Save the result of Stage 1
    with torch.no_grad():
        n_ideal, d_ideal_nm = model(seed_vec)
        T_ideal = forward_spectrum(n_ideal, d_ideal_nm, cfg, 
                                   with_grading=False, 
                                   with_systematic=False, 
                                   with_random=False)
        
        T_ideal_with_errors = forward_spectrum(n_ideal, d_ideal_nm, cfg,
                                               with_grading=True,
                                               with_systematic=True,
                                               with_random=False)
        
        T_ideal_with_all_errors = forward_spectrum(n_ideal, d_ideal_nm, cfg,
                                                   with_grading=True,
                                                   with_systematic=True,
                                                   with_random=True)


    # --- Stage 2: Fabrication-in-the-Loop ---
    if verbose:
        print(f"  Running Stage 2 (Fabrication-in-Loop) for {cfg.iters_stage2} iterations...")
    
    # We re-use the main optimizer for Stage 2
    for it in range(1, cfg.iters_stage2 + 1):
        opt.zero_grad()
        n_hat, d_hat_nm = model(seed_vec)
        
        T_fab = forward_spectrum(n_hat, d_hat_nm, cfg,
                                 with_grading=True,
                                 with_systematic=True,
                                 with_random=True)
        
        loss = mse(T_fab, target_spectrum)
        loss.backward()
        opt.step()


    if verbose:
        print(f"  Fine-tuning Stage 2 for {cfg.iters_finetune} iterations...")
    # Create a new optimizer with a smaller learning rate
    opt_finetune_s2 = optim.Adam(model.parameters(), lr=cfg.lr_finetune)
    for it in range(1, cfg.iters_finetune + 1):
        opt_finetune_s2.zero_grad()
        n_hat, d_hat_nm = model(seed_vec)
        
        T_fab = forward_spectrum(n_hat, d_hat_nm, cfg,
                                 with_grading=True,
                                 with_systematic=True,
                                 with_random=True)
        
        loss = mse(T_fab, target_spectrum)
        loss.backward()
        opt_finetune_s2.step()

    model_filename = f"model_{cfg.num_layers}L_{cfg.sys_mode_n}.pth"
    model_save_path = os.path.join(OUTPUT_DIR, model_filename)
    
    torch.save(model.state_dict(), model_save_path)
    if verbose:
        print(f"  Final model saved to: {model_save_path}")


    # Save the final (compensated) result
    with torch.no_grad():
        n_final, d_final_nm = model(seed_vec)
        
        T_final_compensated = forward_spectrum(n_final, d_final_nm, cfg,
                                               with_grading=True,
                                               with_systematic=True,
                                               with_random=False)
        
        T_final_compensated_all_errors = forward_spectrum(n_final, d_final_nm, cfg,
                                                          with_grading=True,
                                                          with_systematic=True,
                                                          with_random=True)
    
    return {
        "n_ideal": n_ideal.detach(),
        "d_ideal_nm": d_ideal_nm.detach(),
        "T_ideal": T_ideal.detach().cpu().numpy(),
        "T_ideal_with_errors": T_ideal_with_errors.detach().cpu().numpy(),
        "T_ideal_with_all_errors": T_ideal_with_all_errors.detach().cpu().numpy(),
        "n_final": n_final.detach(),
        "d_final_nm": d_final_nm.detach(),
        "T_final_compensated": T_final_compensated.detach().cpu().numpy(),
        "T_final_compensated_all_errors": T_final_compensated_all_errors.detach().cpu().numpy(),
        "saved_model_path": model_save_path
    }


def run_monte_carlo(n_core: torch.Tensor, d_core_nm: torch.Tensor, cfg: DesignConfig, target_spectrum: torch.Tensor) -> Dict[str, Any]:
    """
    Runs Monte Carlo simulation on a *fixed* design stack.
    Returns the spectra for the best (min loss) and worst (max loss) trials.
    """
    target_spectrum = target_spectrum.to(DEVICE)
    
    all_trials_T = []
    all_trials_loss = []
    
    with torch.no_grad():
        for _ in range(cfg.mc_trials):
            # MC eval includes grading, systematic, and random errors
            T_mc = forward_spectrum(n_core, d_core_nm, cfg,
                                    with_grading=True,
                                    with_systematic=True,
                                    with_random=True)
            
            loss = mse(T_mc, target_spectrum)
            all_trials_T.append(T_mc.cpu().numpy())
            all_trials_loss.append(loss.item())
            
    best_idx = np.argmin(all_trials_loss)
    worst_idx = np.argmax(all_trials_loss)
    
    return {
        "T_best": all_trials_T[best_idx],
        "T_worst": all_trials_T[worst_idx],
        "losses": all_trials_loss
    }


# -----------------------------
# 7) Functions to Generate Each Figure
# -----------------------------

def generate_figure_3_3(base_cfg: DesignConfig):
    """
    Experiment 3.3: Optimization for various targets and layer counts.
    (a-f) shows 6 different filter designs.
    """
    print("\n--- Performing Experiment 3.3 ---")
    
    # --- Define Targets ---
    wl_nm = WAVELENGTHS_NM.cpu().numpy()
    
    # (a) Long pass filter, 50 layers
    target_a = torch.ones_like(WAVELENGTHS_NM)
    target_a[WAVELENGTHS_NM < 475.0] = 0.0
    
    # (b) Long pass filter, 200 layers
    target_b = target_a.clone()
    
    # (c) Dual stop band, 50 layers
    target_c = torch.ones_like(WAVELENGTHS_NM)
    target_c[(WAVELENGTHS_NM > 437.5) & (WAVELENGTHS_NM < 475.0)] = 0.0
    target_c[(WAVELENGTHS_NM > 587.5) & (WAVELENGTHS_NM < 625.0)] = 0.0
    
    # (d) Dual stop band, 200 layers
    target_d = target_c.clone()

    # (e) Triple stop band, 50 layers
    target_e = torch.ones_like(WAVELENGTHS_NM)
    target_e[(WAVELENGTHS_NM > 420) & (WAVELENGTHS_NM < 450)] = 0.0
    target_e[(WAVELENGTHS_NM > 520) & (WAVELENGTHS_NM < 550)] = 0.0
    target_e[(WAVELENGTHS_NM > 620) & (WAVELENGTHS_NM < 650)] = 0.0

    # (f) Triple stop band, 200 layers
    target_f = target_e.clone()

    specs = {
        'a': {'target': target_a, 'layers': 50},
        'b': {'target': target_b, 'layers': 200},
        'c': {'target': target_c, 'layers': 50},
        'd': {'target': target_d, 'layers': 200},
        'e': {'target': target_e, 'layers': 50},
        'f': {'target': target_f, 'layers': 200},
    }
    
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    axes = axes.flatten()
    
    for i, (key, spec) in enumerate(specs.items()):
        print(f"  Running optimization for plot (3.3{key})...")
        cfg = replace(base_cfg, num_layers=spec['layers'])
        
        # For this plot, we only care about the ideal (Stage 1) result
        # We run the full pipeline but only use the 'T_ideal' output
        results = run_optimization(cfg, spec['target'], verbose=False)
        
        T_optimized = results['T_ideal']
        T_target = spec['target'].cpu().numpy()
        
        ax = axes[i]
        ax.plot(wl_nm, T_optimized, 'b-', label="Optimized Spectra")
        ax.plot(wl_nm, T_target, 'darkorange', label="Target Spectra")
        ax.set_title(f"({key}) Target, {spec['layers']} Layers")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Transmittance")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle=':', alpha=0.6)
    
    axes[0].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_3_3.png"), dpi=150)
    print("  Figure 3.3 saved.")


def generate_figure_3_3_with_stacks(base_cfg: DesignConfig):
    """
    Experiment 3.3 with refractive index stack visualizations.
    """
    print("\n--- Performing Experiment 3.3 (With Index Stacks) ---")
    
    # --- Define Targets ---
    wl_nm = WAVELENGTHS_NM.cpu().numpy()
    
    # 1. Long Pass (a, b)
    target_lp = torch.ones_like(WAVELENGTHS_NM)
    target_lp[WAVELENGTHS_NM < 475.0] = 0.0
    
    # 2. Dual Stop (c, d)
    target_ds = torch.ones_like(WAVELENGTHS_NM)
    target_ds[(WAVELENGTHS_NM > 437.5) & (WAVELENGTHS_NM < 475.0)] = 0.0
    target_ds[(WAVELENGTHS_NM > 587.5) & (WAVELENGTHS_NM < 625.0)] = 0.0

    # 3. Triple Stop (e, f) - Approximated from image visual
    target_ts = torch.ones_like(WAVELENGTHS_NM)
    target_ts[(WAVELENGTHS_NM > 430) & (WAVELENGTHS_NM < 460)] = 0.0
    target_ts[(WAVELENGTHS_NM > 520) & (WAVELENGTHS_NM < 560)] = 0.0
    target_ts[(WAVELENGTHS_NM > 620) & (WAVELENGTHS_NM < 650)] = 0.0

    specs = {
        'a': {'target': target_lp, 'layers': 50},
        'b': {'target': target_lp, 'layers': 200},
        'c': {'target': target_ds, 'layers': 50},
        'd': {'target': target_ds, 'layers': 200},
        'e': {'target': target_ts, 'layers': 50},
        'f': {'target': target_ts, 'layers': 200},
    }
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    # Global colormap for indices
    cmap = mpl.colormaps['seismic']
    norm = mcolors.Normalize(vmin=1.6, vmax=2.4)

    for i, (key, spec) in enumerate(specs.items()):
        print(f"  Optimizing (3.3{key})...")
        cfg = replace(base_cfg, num_layers=spec['layers'])
        results = run_optimization(cfg, spec['target'], verbose=False)
        
        T_optimized = results['T_ideal']
        T_target = spec['target'].cpu().numpy()
        n_final = results['n_ideal'].cpu().numpy() # Get the indices
        
        ax = axes[i]
        
        # 1. Plot Spectra
        ax.plot(wl_nm, T_optimized, 'C0', label="Optimized Spectra", linewidth=1.5)
        ax.plot(wl_nm, T_target, 'C1', label="Target Spectra", linewidth=1.5, alpha=0.9)
        
        # 2. Add the Stack Bar
        if i % 2 == 0: # Left column (a, c, e) -> Bar on Left
            inset_ax = ax.inset_axes([0.15, 0.1, 0.15, 0.8])
        else:          # Right column (b, d, f) -> Bar on Right
            inset_ax = ax.inset_axes([0.80, 0.1, 0.15, 0.8])
            
        inset_ax.imshow(n_final.reshape(-1, 1), aspect='auto', cmap=cmap, norm=norm, origin='lower')
        
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
        for spine in inset_ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(0.5)

        # 3. Formatting
        ax.set_title(f"({key})", loc='left', fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle=':', alpha=0.5)
        
        if i >= 4: ax.set_xlabel("Wavelength (nm)", fontsize=12)
        if i % 2 == 0: ax.set_ylabel("Transmittance", fontsize=12)

    # Add a colorbar for the index (1.6 - 2.4)
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.4, 0.015, 0.2]) 
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cb.set_ticks([1.6, 1.8, 2.0, 2.2, 2.4])
    cb.set_label("Refractive Index", rotation=270, labelpad=15)

    axes[0].legend(loc="lower center", fontsize=10)

    # ** Use this corrected layout call **
    fig.tight_layout(rect=[0, 0, 0.9, 1]) # Make room for colorbar
    
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_3_3_with_stacks.png"), dpi=150)
    print("  Figure 3.3 (with stacks) saved.")

def generate_figure_3_4(base_cfg: DesignConfig):
    """
    Experiment 3.4: Compares 50 vs 200 layers for two filter types.
    This is a subset of Figure 3.3.
    """
    print("\n--- Performing Experiment 3.4 ---")
    
    # --- Define Targets ---
    wl_nm = WAVELENGTHS_NM.cpu().numpy()
    
    # (a, b) Long pass filter
    target_lp = torch.ones_like(WAVELENGTHS_NM)
    target_lp[WAVELENGTHS_NM < 475.0] = 0.0
    
    # (c, d) Dual stop band
    target_ds = torch.ones_like(WAVELENGTHS_NM)
    target_ds[(WAVELENGTHS_NM > 437.5) & (WAVELENGTHS_NM < 475.0)] = 0.0
    target_ds[(WAVELENGTHS_NM > 587.5) & (WAVELENGTHS_NM < 625.0)] = 0.0

    specs = {
        'a': {'target': target_lp, 'layers': 50},
        'b': {'target': target_lp, 'layers': 200},
        'c': {'target': target_ds, 'layers': 50},
        'd': {'target': target_ds, 'layers': 200},
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    for i, (key, spec) in enumerate(specs.items()):
        print(f"  Running optimization for plot (3.4{key})...")
        cfg = replace(base_cfg, num_layers=spec['layers'])
        
        # We only care about the ideal (Stage 1) result
        results = run_optimization(cfg, spec['target'], verbose=False)
        
        T_optimized = results['T_ideal']
        T_target = spec['target'].cpu().numpy()
        
        ax = axes[i]
        ax.plot(wl_nm, T_optimized, 'b-', label="Optimized Spectra")
        ax.plot(wl_nm, T_target, 'darkorange', label="Target Spectra")
        ax.set_title(f"({key}) {spec['layers']} Layers")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Transmittance")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle=':', alpha=0.6)
    
    axes[0].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_3_4.png"), dpi=150)
    print("  Figure 3.4 saved.")


def generate_figure_3_5(base_cfg: DesignConfig):
    """
    Experiment 3.5: Compares performance for 300, 200, and 130 layers.
    UPDATED: Increases iterations for this specific experiment to ensure
    the 300-layer baseline converges to a flat, smooth stopband.
    """
    print("\n--- Performing Experiment 3.5 (Layer Reduction Validation) ---")
    
    # Define a single band-stop target (Notch at 550nm)
    # We use the global WAVELENGTHS_NM tensor
    target = torch.ones_like(WAVELENGTHS_NM)
    target[(WAVELENGTHS_NM > 530) & (WAVELENGTHS_NM < 570)] = 0.0 
    
    layer_counts = [300, 200, 130]
    results_T = {}
    
    for layers in layer_counts:
        print(f"  Running optimization for {layers} layers...")
        
        # CREATE A BEEFED-UP CONFIGURATION
        # 1. Increase iterations significantly (300 layers needs more time to settle)
        # 2. Lower the learning rate slightly to avoid jitter in the deep stopband
        cfg = replace(base_cfg, 
                      num_layers=layers,
                      iters_stage1=10000,       # Increased from default 800
                      iters_finetune=2000,     # Increased finetuning
                      lr=1e-3                  # Standard start
                      )
        
        # Run optimization
        # We only care about the 'T_ideal' (Stage 1 result) for this physics validation
        results = run_optimization(cfg, target, verbose=False)
        results_T[layers] = results['T_ideal'] 
        
    # Plotting
    wl_nm = WAVELENGTHS_NM.cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot (a): 300 vs 130
    axes[0].plot(wl_nm, results_T[300], color='darkorange', linewidth=1.5, label="300 Layers (Baseline)")
    axes[0].plot(wl_nm, results_T[130], color='blue', linewidth=1.5, label="130 Layers")
    axes[0].set_title("(a) 300 vs 130 Layers")
    axes[0].set_xlabel("Wavelength (nm)")
    axes[0].set_ylabel("Transmittance")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend(loc='lower left')
    
    # Plot (b): 300 vs 200
    axes[1].plot(wl_nm, results_T[300], color='darkorange', linewidth=1.5, label="300 Layers (Baseline)")
    axes[1].plot(wl_nm, results_T[200], color='blue', linewidth=1.5, label="200 Layers")
    axes[1].set_title("(b) 300 vs 200 Layers")
    axes[1].set_xlabel("Wavelength (nm)")
    axes[1].set_ylabel("Transmittance")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend(loc='lower left')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "fig_3_5_10000iters.png")
    plt.savefig(save_path, dpi=150)
    print(f"  Figure 3.5 saved to {save_path}")


def generate_figure_3_7(base_cfg: DesignConfig):
    """
    Experiment 3.7: Optimization under DETERMINISTIC imperfections.
    - Blue: Ideal stack, ideal eval
    - Orange: Ideal stack, error eval
    - Black: Compensated stack, error eval
    """
    print("\n--- Performing Experiment 3.7 ---")
    
    # Define a single band-stop target
    wl_nm = WAVELENGTHS_NM.cpu().numpy()
    target = torch.ones_like(WAVELENGTHS_NM)
    target[(WAVELENGTHS_NM > 530) & (WAVELENGTHS_NM < 570)] = 0.0

    # --- Define Error Configs ---
    # (a) Linear gradient shift
    cfg_a = replace(base_cfg, 
                    sys_mode_n="linear", 
                    sys_params_n={"a": 0.4}) # 0.4 shift over the whole stack
    
    # (b) Graded transition regions (already in base_cfg, just no other errors)
    cfg_b = base_cfg # Assumes base_cfg has grade_nm > 0
    
    # (c) Combined linear + graded
    cfg_c = cfg_a # Already has linear + grading
    
    # (d) Polynomial
    cfg_d = replace(base_cfg, 
                    sys_mode_n="poly", 
                    sys_params_n={"a": 0.3, "p": 3.0}) # 0.3*z^3
    
    # (e) Sinusoidal
    cfg_e = replace(base_cfg, 
                    sys_mode_n="sine", 
                    sys_params_n={"a": 0.2, "f": 3.0}) # 0.2*sin(2*pi*3*z)
    
    # (f) Exponential
    cfg_f = replace(base_cfg, 
                    sys_mode_n="exp", 
                    sys_params_n={"a": 0.3, "b": 4.0}) # 0.3*exp(4z)

    configs = {'a': cfg_a, 'b': cfg_b, 'c': cfg_c, 'd': cfg_d, 'e': cfg_e, 'f': cfg_f}

    plot_titles = {
        'a': "(a) Linear gradient shift",
        'b': "(b) Graded transition regions",
        'c': "(c) Combined linear + graded",
        'd': "(d) Polynomial shift",
        'e': "(e) Sinusoidal shift",
        'f': "(f) Exponential shift"
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    
    for i, (key, cfg) in enumerate(configs.items()):
        print(f"  Running optimization for plot (3.7{key})...")
        
        # We need the full optimization results
        results = run_optimization(cfg, target, verbose=False)

        # T_blue: "original, ideally optimized stack"
        # This is the T_ideal from the run
        T_blue = results['T_ideal']
        
        # T_orange: "same stack after introducing fabrication errors"
        # This is the T_ideal_with_errors from the run
        T_orange = results['T_ideal_with_errors']
        
        # T_black: "spectra after completing the second stage"
        # This is the T_final_compensated from the run
        T_black = results['T_final_compensated']

        ax = axes[i]
        ax.plot(wl_nm, T_blue, 'b-', label="Ideal")
        ax.plot(wl_nm, T_orange, 'darkorange', label="Ideal w/ Error")
        ax.plot(wl_nm, T_black, 'k,', linestyle='--', label="Compensated") # Dotted black
        
        ax.set_title(plot_titles[key])
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Transmittance")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle=':', alpha=0.6)
    
    axes[0].legend()
    plt.xlim(400, 700)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_3_7.png"), dpi=150)
    print("  Figure 3.7 saved.")


def generate_figure_3_9(base_cfg: DesignConfig):
    """
    Experiment 3.9: Optimization in presence of RANDOM imperfections.
    Compares an ideally-optimized stack vs. a random-aware stack,
    when both are subjected to 1000 MC trials.
    Shows the best and worst performers from the trials.
    """
    print("\n--- Performing Experiment 3.9 ---")
    
    # Define a single band-stop target
    wl_nm = WAVELENGTHS_NM.cpu().numpy()
    target = torch.ones_like(WAVELENGTHS_NM)
    target[(WAVELENGTHS_NM > 530) & (WAVELENGTHS_NM < 570)] = 0.0

    # --- Define Error Configs ---
    # (a, e) µ=0%, γ=±1%
    cfg_ae = replace(base_cfg, rnd_n_mu=0.0, rnd_n_gamma=0.01)
    
    # (b, f) µ=1%, γ=±1%
    cfg_bf = replace(base_cfg, rnd_n_mu=0.01, rnd_n_gamma=0.01)
    
    # (c, g) µ=0%, γ=±5%
    cfg_cg = replace(base_cfg, rnd_n_mu=0.0, rnd_n_gamma=0.05)

    # (d, h) µ=5%, γ=±5%
    cfg_dh = replace(base_cfg, rnd_n_mu=0.05, rnd_n_gamma=0.05)
    
    configs = {'ae': cfg_ae, 'bf': cfg_bf, 'cg': cfg_cg, 'dh': cfg_dh}
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    
    for i, (key, cfg) in enumerate(configs.items()):
        print(f"  Running optimization for plot (3.9{key})...")
        
        # --- Run 1: Ideal-only optimization ---
        # No errors during training
        cfg_ideal_train = replace(cfg, 
                                  rnd_n_mu=0.0, rnd_n_gamma=0.0, 
                                  iters_stage2=1) # No stage 2
        ideal_results = run_optimization(cfg_ideal_train, target, verbose=False)
        n_ideal, d_ideal_nm = ideal_results['n_ideal'], ideal_results['d_ideal_nm']

        # --- Run 2: Full random-aware optimization ---
        # Stage 2 *includes* random errors
        full_results = run_optimization(cfg, target, verbose=False)
        n_final, d_final_nm = full_results['n_final'], full_results['d_final_nm']

        # --- MC Eval ---
        print(f"    Running MC for (3.9{key}) Ideal Stack...")
        mc_ideal = run_monte_carlo(n_ideal, d_ideal_nm, cfg, target)
        
        print(f"    Running MC for (3.9{key}) Compensated Stack...")
        mc_comp = run_monte_carlo(n_final, d_final_nm, cfg, target)

        # --- Plotting ---
        # Top row: Best performers (min loss)
        ax_top = axes[0, i]
        ax_top.plot(wl_nm, target.cpu().numpy(), 'darkorange', label="Target")
        ax_top.plot(wl_nm, mc_ideal['T_best'], 'b-', label="Ideal (Best Trial)")
        ax_top.plot(wl_nm, mc_comp['T_best'], 'k,', linestyle='--', label="Comp (Best Trial)")
        ax_top.set_title(f"({key[0]}) Best Trial (µ={cfg.rnd_n_mu*100}%, γ=±{cfg.rnd_n_gamma*100}%)")
        ax_top.set_ylabel("Transmittance")
        ax_top.set_ylim(-0.05, 1.05)
        
        # Bottom row: Worst performers (max loss)
        ax_bot = axes[1, i]
        ax_bot.plot(wl_nm, target.cpu().numpy(), 'darkorange', label="Target")
        ax_bot.plot(wl_nm, mc_ideal['T_worst'], 'b-', label="Ideal (Worst Trial)")
        ax_bot.plot(wl_nm, mc_comp['T_worst'], 'k,', linestyle='--', label="Comp (Worst Trial)")
        ax_bot.set_title(f"({key[1]}) Worst Trial (µ={cfg.rnd_n_mu*100}%, γ=±{cfg.rnd_n_gamma*100}%)")
        ax_bot.set_xlabel("Wavelength (nm)")
        ax_bot.set_ylabel("Transmittance")
        ax_bot.set_ylim(-0.05, 1.05)

    axes[0, 0].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_3_9.png"), dpi=150)
    print("  Figure 3.9 saved.")


def generate_figure_3_10(base_cfg: DesignConfig):
    """
    Experiment 3.10: Both DETERMINISTIC and RANDOM imperfections.
    - Blue: Ideal stack, ideal eval
    - Orange: Ideal stack, eval with *both* error types
    - Black: Compensated stack, eval with *both* error types
    """
    print("\n--- Performing Experiment 3.10 ---")
    
    # Define a single band-stop target
    wl_nm = WAVELENGTHS_NM.cpu().numpy()
    target = torch.ones_like(WAVELENGTHS_NM)
    target[(WAVELENGTHS_NM > 530) & (WAVELENGTHS_NM < 570)] = 0.0

    # --- Define Error Configs ---
    # (a) Linear + Graded + γ=±1%
    cfg_a = replace(base_cfg, 
                    sys_mode_n="linear", 
                    sys_params_n={"a": 0.4},
                    rnd_n_gamma=0.01)
    
    # (b) Linear + Graded + γ=±5%
    cfg_b = replace(base_cfg, 
                    sys_mode_n="linear", 
                    sys_params_n={"a": 0.4},
                    rnd_n_gamma=0.05)

    # (c) Sinusoidal + γ=±1%
    cfg_c = replace(base_cfg, 
                    sys_mode_n="sine", 
                    sys_params_n={"a": 0.2, "f": 3.0},
                    rnd_n_gamma=0.01)
    
    # (d) Sinusoidal + γ=±5%
    cfg_d = replace(base_cfg, 
                    sys_mode_n="sine", 
                    sys_params_n={"a": 0.2, "f": 3.0},
                    rnd_n_gamma=0.05)

    configs = {'a': cfg_a, 'b': cfg_b, 'c': cfg_c, 'd': cfg_d}

    plot_titles = {
        'a': "(a) Linear + Graded + γ=±1%",
        'b': "(b) Linear + Graded + γ=±5%",
        'c': "(c) Sinusoidal + γ=±1%",
        'd': "(d) Sinusoidal + γ=±5%"
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    for i, (key, cfg) in enumerate(configs.items()):
        print(f"  Running optimization for plot (3.10{key})...")
        
        # We need the full optimization results
        results = run_optimization(cfg, target, verbose=False)

        # T_blue: "ideally optimized spectra without any error"
        T_blue = results['T_ideal']
        
        # T_orange: "ideally optimized spectra after both error types"
        # This is the T_ideal_with_all_errors (Det + 1x Rand)
        T_orange = results['T_ideal_with_all_errors']
        
        # T_black: "result of optimization to compensate"
        # This is T_final_compensated_all_errors (Det + 1x Rand)
        T_black = results['T_final_compensated_all_errors']

        ax = axes[i]
        ax.plot(wl_nm, T_blue, 'b-', label="Ideal")
        ax.plot(wl_nm, T_orange, 'darkorange', label="Ideal w/ Errors (Det+Rand)")
        ax.plot(wl_nm, T_black, 'k,', linestyle='--', label="Compensated (Det+Rand)")
        
        ax.set_title(plot_titles[key])
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Transmittance")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle=':', alpha=0.6)
    
    axes[0].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_3_10.png"), dpi=150)
    print("  Figure 3.10 saved.")


# -----------------------------
# 8) Main execution block
# -----------------------------
if __name__ == "__main__":
    
    print(f"Starting the experiment. All outputs will be saved to '{OUTPUT_DIR}/'")
    print("="*60)
    print("!!! WARNING !!!")
    print("This script will run MANY (10+) full optimizations and Monte Carlo simulations.")
    print(f"Iterations: {DesignConfig.iters_stage1} and trials have been set to {DesignConfig.mc_trials} iters.")
    print("="*60)
    
    start_time = time.time()
    
    # Create a base configuration. Specific figures will modify this.
    base_config = DesignConfig(
        grade_nm=8.0,       # Include grading by default as in Fig 3.7+
        grade_step_nm=4.0
    )

    # --- Run functions for each figure ---

    # generate_figure_3_3_with_stacks(base_config)

    # generate_figure_3_3(base_config)
    
    # generate_figure_3_4(base_config)
    
    generate_figure_3_5(base_config)
    
    # generate_figure_3_7(base_config)
    
    # generate_figure_3_9(base_config)

    # generate_figure_3_10(base_config)

    
    end_time = time.time()
    print("\n" + "="*60)
    print(f"All tasks complete. Total time: {end_time - start_time:.2f} seconds.")
    print(f"Figures saved in '{OUTPUT_DIR}/'")
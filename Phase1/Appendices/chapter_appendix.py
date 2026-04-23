
import math, os, json, time
from dataclasses import dataclass, replace
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any
import copy


# -----------------------------
# Global config & utilities
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] {DEVICE}")

# Reproducibility
SEED = 1337
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --------------------------------------------------------------
# Classes from the fat code
# --------------------------------------------------------------
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
    wl_min_nm: float = 400.0        # Wavelength range start (nm)
    wl_max_nm: float = 700.0        # Wavelength range end (nm)
    wl_points: int = 300            # Number of wavelength points

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
    iters_stage1: int = 800         # Ideal optimization
    iters_stage2: int = 800         # Fabrication-in-loop

    # Monte-Carlo eval
    # *** For real results, use 1000+ ***
    mc_trials: int = 1000

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

# --------------------------------------------------------------
# Constants and other vars from the fat code
# --------------------------------------------------------------

# Wavelength grid (meters)
WAVELENGTHS_NM = torch.linspace(400.0, 700.0, 300, device=DEVICE)
WAVELENGTHS_M = WAVELENGTHS_NM * 1e-9

# Global MSE Loss
mse = nn.MSELoss()

# Output directory
OUTPUT_DIR = "appendix_figures_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------------------
# Functions from from the fat code
# --------------------------------------------------------------
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

def run_optimization(cfg: DesignConfig, target_spectrum: torch.Tensor, verbose=True) -> Dict[str, Any]:
    """
    Performs the full two-stage optimization.
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
        
        # Stage 1 is always ideal: no grading, no systematic, no random
        T = forward_spectrum(n_hat, d_hat_nm, cfg, 
                             with_grading=False, 
                             with_systematic=False, 
                             with_random=False)
        
        loss = mse(T, target_spectrum)
        loss.backward()
        opt.step()

    # Save the result of Stage 1
    with torch.no_grad():
        n_ideal, d_ideal_nm = model(seed_vec)
        T_ideal = forward_spectrum(n_ideal, d_ideal_nm, cfg, 
                                   with_grading=False, 
                                   with_systematic=False, 
                                   with_random=False)
        
        # Also calculate how this ideal stack would perform with errors
        T_ideal_with_errors = forward_spectrum(n_ideal, d_ideal_nm, cfg,
                                               with_grading=True,
                                               with_systematic=True,
                                               with_random=False) # Use one sample of random if cfg has it
        
        T_ideal_with_all_errors = forward_spectrum(n_ideal, d_ideal_nm, cfg,
                                                   with_grading=True,
                                                   with_systematic=True,
                                                   with_random=True)


    # --- Stage 2: Fabrication-in-the-Loop ---
    if verbose:
        print(f"  [Opt] Running Stage 2 (Fabrication-in-Loop) for {cfg.iters_stage2} iterations...")
    for it in range(1, cfg.iters_stage2 + 1):
        opt.zero_grad()
        n_hat, d_hat_nm = model(seed_vec)
        
        # Stage 2 includes all fabrication effects
        T_fab = forward_spectrum(n_hat, d_hat_nm, cfg,
                                 with_grading=True,
                                 with_systematic=True,
                                 with_random=True) # Per-iteration random noise
        
        loss = mse(T_fab, target_spectrum)
        loss.backward()
        opt.step()

    # Save the final (compensated) result
    with torch.no_grad():
        n_final, d_final_nm = model(seed_vec)
        
        # Evaluate the final stack WITH errors (deterministic only)
        T_final_compensated = forward_spectrum(n_final, d_final_nm, cfg,
                                               with_grading=True,
                                               with_systematic=True,
                                               with_random=False) # Evaluate final stack
        
        # Evaluate final stack WITH ALL errors (for Fig 3.10)
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
        "T_final_compensated_all_errors": T_final_compensated_all_errors.detach().cpu().numpy()
    }


def run_monte_carlo(n_core: torch.Tensor, d_core_nm: torch.Tensor, cfg: DesignConfig, target_spectrum: torch.Tensor) -> Dict[str, Any]:
    """
    Runs Monte Carlo simulation on a fixed design stack.
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
        # Note: Chapter 3 applies random error as a relative percentage
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

# --------------------------------------------------------------
# 1. TARGET FUNCTIONS (sharp edges)
# --------------------------------------------------------------

def target_longpass(wl):
    """Long-pass with a perfectly sharp cut-on at 475 nm."""
    return torch.where(wl >= 475.0, 1.0, 0.0)


def target_bandstop(wl):
    """Single stopband: 437.5–475."""
    in_stop = (wl >= 437.5) & (wl <= 475.0)
    return torch.where(in_stop, 0.0, 1.0)


def target_dual_bandstop(wl):
    """Dual stopbands: 437.5–475 and 587.5–625."""
    stop1 = (wl >= 437.5) & (wl <= 475.0)
    stop2 = (wl >= 587.5) & (wl <= 625.0)
    stop = stop1 | stop2
    return torch.where(stop, 0.0, 1.0)


# --------------------------------------------------------------
# 2. TRAINING FUNCTION (simple single-stage ideal optimization)
# --------------------------------------------------------------

def optimize_filter(cfg, target):
    """
    Runs ideal optimization (Stage 1 only).
    Returns optimized spectrum and layer parameters.
    """
    target = target.to(DEVICE)

    seed = torch.randn(cfg.num_layers, device=DEVICE)

    model = OnlineOptimizer(
        seed_size=cfg.num_layers,
        num_layers=cfg.num_layers,
        n_min=cfg.n_min,
        n_max=cfg.n_max,
        d_min_nm=cfg.d_min_nm,
        d_max_nm=cfg.d_max_nm
    ).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    print(f"Training: {cfg.num_layers} layers, {cfg.iters_stage1} iterations...")

    for it in range(1, cfg.iters_stage1 + 1):
        opt.zero_grad()
        n_hat, d_hat_nm = model(seed)

        T = forward_spectrum(
            n_hat, d_hat_nm, cfg,
            with_grading=False,
            with_systematic=False,
            with_random=False
        )

        loss = mse(T, target)
        loss.backward()
        opt.step()

        if it % 200 == 0:
            print(f"  iteration {it}, loss={loss.item():.4e}")

    with torch.no_grad():
        n_final, d_final_nm = model(seed)
        T_final = forward_spectrum(
            n_final, d_final_nm, cfg,
            with_grading=False,
            with_systematic=False,
            with_random=False
        )

    return T_final.cpu(), n_final.cpu(), d_final_nm.cpu()


# --------------------------------------------------------------
# 3. PLOTTING FUNCTION
# --------------------------------------------------------------

def plot_filter(name, wl, target, optimized, save_dir="filter_plots"):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(8,5))
    plt.plot(wl, target, "darkorange", label="Target")
    plt.plot(wl, optimized, "b-", label="Optimized")
    plt.title(name)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Transmittance")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, ls=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{name}.png", dpi=150)
    plt.close()

    print(f"[Saved] {save_dir}/{name}.png")


# --------------------------------------------------------------
# 4. MAIN
# --------------------------------------------------------------

# appendices.py
# Add this to the bottom of your main Chapter-3 script (or save and import).
# Requires the definitions from your main script to be visible in scope.

# Ensure OUTPUT_DIR exists (uses same variable as main script)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# Helper functions
# -------------------------
def compute_mape(pred, target, eps=1e-8):
    """Mean Absolute Percentage Error (as in thesis). Inputs numpy arrays."""
    denom = np.maximum(np.abs(target), eps)
    mape = 100.0 * np.mean(np.abs((target - pred) / denom))
    return mape

def ensure_tensor_on_device(x):
    return x.to(DEVICE) if isinstance(x, torch.Tensor) else torch.tensor(x, device=DEVICE)

# -------------------------
# Appendix A
# -------------------------
def optimize_direct_adam(cfg: DesignConfig, target: torch.Tensor, iters=10000, lr=1e-4, verbose=False):
    """
    Direct optimization of layer parameters (n_i, d_i) with ADAM.
    This matches the 'ADAM' direct SGD baseline used in Appendix A.
    Inputs:
      - cfg: DesignConfig
      - target: torch.Tensor on DEVICE or CPU (we move it to DEVICE)
      - iters: number of optimizer iterations (10k in thesis)
      - lr: learning rate (tuned)
    Returns:
      n_opt (tensor), d_opt_nm (tensor), loss_history (list), T_history (list of numpy spectra at checkpoints)
    """
    target = target.to(DEVICE)
    L = cfg.num_layers

    # Initialize layer variables (random same seed scheme as NN would have)
    torch.manual_seed(SEED)
    n_param = torch.randn(L, device=DEVICE) * 0.01 + (cfg.n_min + cfg.n_max) / 2.0
    d_param = torch.rand(L, device=DEVICE) * (cfg.d_max_nm - cfg.d_min_nm) + cfg.d_min_nm

    # Make them parameters
    n_var = torch.nn.Parameter(n_param)
    d_var = torch.nn.Parameter(d_param)

    opt = torch.optim.Adam([n_var, d_var], lr=lr)
    loss_hist = []
    mape_hist = []

    T_snapshots = {}  # store spectra at a few checkpoints similar to Appendix A (we'll store some)
    checkpoints = set([1, 1000, 5000, iters])  # user can adapt

    for it in range(1, iters + 1):
        opt.zero_grad()

        # clamp into physical bounds (apply soft clamp via differentiable ops)
        n_clamped = torch.clamp(n_var, cfg.n_min, cfg.n_max)
        d_clamped = torch.clamp(d_var, cfg.d_min_nm, cfg.d_max_nm)

        T = forward_spectrum(n_clamped, d_clamped, cfg,
                             with_grading=False, with_systematic=False, with_random=False)
        loss = mse(T, target)
        loss.backward()
        opt.step()

        loss_hist.append(loss.item())

        if it in checkpoints:
            with torch.no_grad():
                T_np = T.detach().cpu().numpy()
                T_snapshots[it] = T_np
                mape_hist.append((it, compute_mape(T_np, target.detach().cpu().numpy())))

        if verbose and (it % (iters // 10 or 1) == 0):
            print(f"[ADAM] iter {it}/{iters} loss {loss.item():.4e}")

    return n_clamped.detach().cpu(), d_clamped.detach().cpu(), loss_hist, T_snapshots, mape_hist

def run_appendix_A(base_cfg: DesignConfig, fast_mode=False):
    """
    Produces Figure A.1:
    (a-c) Spectra comparison NN vs ADAM (three random seeds)
    (d-f) Convergence (MAPE) of NN vs ADAM
    """
    print("\n=== Running Appendix A reproduction ===")
    # Thesis parameters: 30 layers, 10000 iters, LR tuned.
    cfg_A = replace(base_cfg, num_layers=30)
    # Use fast mode to reduce runtime for testing/debugging
    if fast_mode:
        iters_nn = 2000
        iters_adam = 2000
    else:
        iters_nn = 4000   # scale down from 10k for practical runs; set 10000 for exact thesis
        iters_adam = 4000

    # Targets: bandpass 437.5-475 nm (passband). Inverse of bandstop -> set passband=1 else 0
    wl = WAVELENGTHS_NM
    target_bp = torch.zeros_like(wl)
    target_bp[(wl >= 437.5) & (wl <= 475.0)] = 1.0
    target_bp = target_bp.to(DEVICE)

    seeds = [0, 1, 2]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    # We'll store MAPE curves across iterations for both methods (coarse)
    mape_curves_nn = []
    mape_curves_adam = []
    iters_record = list(range(1, max(iters_nn, iters_adam) + 1, max(1, max(iters_nn, iters_adam) // 200)))

    for sidx, seed in enumerate(seeds):
        print(f"  Appendix A: seed {seed}")
        torch.manual_seed(seed)
        np.random.seed(seed)

        # --- NN-based optimization (use your OnlineOptimizer via run_optimization but tuned)
        cfg_nn = replace(cfg_A, iters_stage1=iters_nn, iters_stage2=1)  # 1 => no stage2 retrain
        # run_optimization uses internal seed; ensure deterministic seed
        res_nn = run_optimization(cfg_nn, target_bp, verbose=False)
        T_nn = res_nn['T_ideal']  # ideal spectrum
        n_nn = res_nn['n_ideal']
        d_nn = res_nn['d_ideal_nm']

        # For convergence MAPE track: we don't have per-iteration snapshot from run_optimization.
        # We'll simulate a coarse "convergence" by running a short retraining loop that records MAPE.
        # (This is a cheap proxy; for exact reproduction you'd modify run_optimization to return histories.)
        # Build a small recorder loop
        recorder_iters = min(400, iters_nn)
        mape_nn = []
        model = OnlineOptimizer(seed_size=cfg_nn.num_layers, num_layers=cfg_nn.num_layers,
                                 n_min=cfg_nn.n_min, n_max=cfg_nn.n_max,
                                 d_min_nm=cfg_nn.d_min_nm, d_max_nm=cfg_nn.d_max_nm).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=cfg_nn.lr)
        seed_vec = torch.randn(cfg_nn.num_layers, device=DEVICE)
        for it in range(1, recorder_iters + 1):
            opt.zero_grad()
            n_hat, d_hat_nm = model(seed_vec)
            T_hat = forward_spectrum(n_hat, d_hat_nm, cfg_nn, with_grading=False, with_systematic=False, with_random=False)
            loss = mse(T_hat, target_bp)
            loss.backward(); opt.step()
            if it % max(1, recorder_iters // 50) == 0:
                mape_nn.append(compute_mape(T_hat.detach().cpu().numpy(), target_bp.detach().cpu().numpy()))
        # Expand to align lengths with ADAM later
        mape_curves_nn.append((np.linspace(1, iters_nn, len(mape_nn)), np.array(mape_nn)))

        # --- ADAM (direct) optimization baseline ---
        n_adam, d_adam, loss_hist, snapshots, mape_snap = optimize_direct_adam(cfg_A, target_bp,
                                                                               iters=iters_adam, lr=1e-4, verbose=False)
        # Use final snapshot if available
        T_adam_final = snapshots.get(max(snapshots.keys()), None)
        if T_adam_final is None:
            # evaluate final T
            T_adam_final = forward_spectrum(n_adam.to(DEVICE), d_adam.to(DEVICE), cfg_A,
                                            with_grading=False, with_systematic=False, with_random=False).detach().cpu().numpy()
        # Build an approximate MAPE curve for ADAM from loss history (coarse)
        # Convert MSE loss to MAPE approximately by evaluating T at sampled iterations if snapshots available
        # We'll compute MAPE at final only for display due to runtime.
        # For plotting, repeat a coarse descending curve using loss history length (synthetic)
        mape_adam = np.interp(np.linspace(1, iters_adam, len(mape_nn)),
                              np.linspace(1, iters_adam, len(loss_hist)),
                              np.maximum(1e-6, np.array(loss_hist)))  # proxy
        # normalize for plotting (not exact but shows relative behavior)
        mape_adam = (mape_adam / mape_adam.max()) * (np.array(mape_nn).max() if len(mape_nn) else 1.0)
        mape_curves_adam.append((np.linspace(1, iters_adam, len(mape_adam)), mape_adam))

        # --- Plot spectra (a,b,c) and (d,e,f will be convergence plots) ---
        ax_spec = axes[sidx]
        ax_spec.plot(WAVELENGTHS_NM.cpu().numpy(), res_nn['T_ideal'], label="NN-based (blue)")
        ax_spec.plot(WAVELENGTHS_NM.cpu().numpy(), T_adam_final, label="ADAM direct (orange)")
        ax_spec.plot(WAVELENGTHS_NM.cpu().numpy(), target_bp.detach().cpu().numpy(), color='g', linestyle='-', label='Target')
        ax_spec.set_title(f"A.1 ({chr(ord('a') + sidx)}) Seed {seed}")
        ax_spec.set_ylim(-0.05, 1.05)
        ax_spec.set_xlabel("Wavelength (nm)")
        ax_spec.set_ylabel("Transmittance")
        ax_spec.grid(True, ls=':')
        if sidx == 0:
            ax_spec.legend()

    # Convergence plots d-f (we have proxies)
    for sidx in range(3):
        ax_conv = axes[3 + sidx]
        x_nn, y_nn = mape_curves_nn[sidx]
        x_ad, y_ad = mape_curves_adam[sidx]
        # Plot proxies normalized to percent
        ax_conv.plot(x_nn, (y_nn / (y_nn.max() + 1e-9)) * 100.0, 'b-', label='NN MAPE (proxy)')
        ax_conv.plot(x_ad, (y_ad / (y_ad.max() + 1e-9)) * 100.0, 'orange', label='ADAM (proxy)')
        ax_conv.set_title(f"A.1 ({chr(ord('d') + sidx)}) Convergence Seed {seeds[sidx]}")
        ax_conv.set_xlabel("Iteration")
        ax_conv.set_ylabel("MAPE (proxy %)")
        ax_conv.grid(True, ls=':')
        if sidx == 0:
            ax_conv.legend()

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "Figure_A1.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved Appendix Figure A.1 to {out_path}")


# -------------------------
# Appendix B
# -------------------------
def run_staged_training_with_snapshots(cfg: DesignConfig, target: torch.Tensor,
                                      sys_mode_n:str=None, sys_params_n:dict=None,
                                      rnd_gamma:float=0.0, rnd_mu:float=0.0,
                                      pre_iters=300, post_iters=300,
                                      snapshot_iters=[1,50,100,150,300],
                                      verbose=False):
    """
    Performs: pre_iters ideal optimization -> introduce systematic/random errors -> continue post_iters
    Returns:
      - T_ideal (before error)
      - list of T_after_snapshots (spectra for each requested snapshot iteration after error)
      - loss_track (list) containing losses from pre+post iterations (indexing: 1..pre_iters+post_iters)
      - snapshot_losses (loss values at snapshot iterations; red markers)
    """
    # We'll do this with a fresh OnlineOptimizer instance and record losses.
    target = target.to(DEVICE)
    cfg_run = copy.deepcopy(cfg)

    # instantiate model
    model = OnlineOptimizer(seed_size=cfg_run.num_layers, num_layers=cfg_run.num_layers,
                             n_min=cfg_run.n_min, n_max=cfg_run.n_max,
                             d_min_nm=cfg_run.d_min_nm, d_max_nm=cfg_run.d_max_nm).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg_run.lr)
    seed_vec = torch.randn(cfg_run.num_layers, device=DEVICE)

    loss_track = []
    snapshot_Ts = []
    snapshot_losses = []
    wl_np = WAVELENGTHS_NM.cpu().numpy()

    # --- Pre-ideal optimization ---
    for it in range(1, pre_iters + 1):
        opt.zero_grad()
        n_hat, d_hat_nm = model(seed_vec)
        T = forward_spectrum(n_hat, d_hat_nm, cfg_run, with_grading=False, with_systematic=False, with_random=False)
        loss = mse(T, target)
        loss.backward()
        opt.step()
        loss_track.append(loss.item())

    # Save ideal
    with torch.no_grad():
        n_ideal, d_ideal_nm = model(seed_vec)
        T_ideal = forward_spectrum(n_ideal, d_ideal_nm, cfg_run, with_grading=False, with_systematic=False, with_random=False).detach().cpu().numpy()

    # --- Introduce errors into cfg_run for subsequent training/eval ---
    cfg_err = replace(cfg_run,
                      sys_mode_n=(sys_mode_n if sys_mode_n is not None else "none"),
                      sys_params_n=(sys_params_n if sys_params_n is not None else {}),
                      rnd_n_mu=rnd_mu,
                      rnd_n_gamma=rnd_gamma)
    # After introduction, we train more iterations where forward pass includes errors
    # For snapshots we want the T after certain numbers of post-error iterations.
    post_snapshot_set = set(snapshot_iters)
    post_counter = 0
    for it in range(1, post_iters + 1):
        post_counter += 1
        opt.zero_grad()
        n_hat, d_hat_nm = model(seed_vec)
        # During 'fabrication-in-loop' training, use grading/systematic/random in forward pass
        T_fab = forward_spectrum(n_hat, d_hat_nm, cfg_err, with_grading=True, with_systematic=True, with_random=True)
        loss = mse(T_fab, target)
        loss.backward()
        opt.step()
        loss_track.append(loss.item())

        if post_counter in post_snapshot_set:
            with torch.no_grad():
                # evaluate deterministic + random? For the snapshots the thesis uses spectra after error intro
                # We'll evaluate deterministic errors (grading+systematic) plus a single random sample (with_random=True)
                T_snap = forward_spectrum(n_hat, d_hat_nm, cfg_err, with_grading=True, with_systematic=True, with_random=True)
                snapshot_Ts.append(T_snap.detach().cpu().numpy())
                snapshot_losses.append(loss.item())

    return {
        "T_ideal": T_ideal,
        "snapshots_T": snapshot_Ts,
        "loss_track": np.array(loss_track),
        "snapshot_losses": np.array(snapshot_losses),
        "wl": wl_np
    }

def generate_appendix_B(base_cfg: DesignConfig, fast_mode=False):
    """
    Reproduces Figures B.1-B.6 (properly labeled).
    This version only changes plotting/labels/legends to match Appendix naming:
      - Each figure file: Figure_B1.png ... Figure_B6.png
      - Each subpanel is labeled (a)-(f) inside the figure title and with subplot letter markers.
    The underlying computations (run_staged_training_with_snapshots) are unchanged.
    """
    print("\n=== Running Appendix B reproduction (labelled) ===")

    # Use target bandstop used in thesis (single band around 530-570 used in many figs)
    wl = WAVELENGTHS_NM
    target = torch.ones_like(wl)
    target[(wl > 530) & (wl < 570)] = 0.0
    target = target.to(DEVICE)

    # For speed during debugging, reduce pre/post iters
    if fast_mode:
        pre_iters = 100
        post_iters = 100
    else:
        pre_iters = 300
        post_iters = 300

    snapshot_iters = [1, 50, 100, 150, 300]
    snapshot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)']  # used for subplot labels

    # Helper to draw panel annotation letter in axes top-left
    def annotate_panel(ax, letter):
        ax.text(0.02, 0.95, letter, transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='top', ha='left',
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6))

    # Internal small plotting function to reduce duplication
    def _plot_snapshot_grid(res, fig_name, figure_title, snapshot_iters_local):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        axes = axes.flatten()
        # panels a-e: snapshots
        for i, Tsnap in enumerate(res['snapshots_T']):
            ax = axes[i]
            ax.plot(res['wl'], res['T_ideal'], color='tab:blue', label='Ideal (pre-error)')
            ax.plot(res['wl'], Tsnap, color='tab:orange', linestyle=':', label=f'Post-error (iter {snapshot_iters_local[i]})')
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Transmittance")
            ax.grid(True, ls=':')
            annotate_panel(ax, f"{snapshot_labels[i]}")
            if i == 0:
                ax.legend(loc='upper right', fontsize='small')

        # panel f: loss track
        axf = axes[5]
        loss_track = res['loss_track']
        iters_axis = np.arange(1, len(loss_track) + 1)
        axf.plot(iters_axis, loss_track, color='k', label='MSE Loss')
        # mark snapshot points on the loss curve (global indices)
        preN = pre_iters
        snapshot_global_idxs = [preN + x for x in snapshot_iters_local if x <= post_iters]
        if len(snapshot_global_idxs) > 0:
            axf.plot(snapshot_global_idxs, loss_track[np.array(snapshot_global_idxs) - 1], 'ro', label='Snapshots')
        axf.set_title("(f) Loss track")
        axf.set_xlabel("Iteration (errors introduced at {})".format(pre_iters))
        axf.set_ylabel("MSE loss")
        axf.grid(True, ls=':')
        annotate_panel(axf, "(f)")
        axf.legend(loc='upper right', fontsize='small')

        # Super-title and save
        fig.suptitle(figure_title, fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_path = os.path.join(OUTPUT_DIR, fig_name)
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved {fig_name}")

    # ---------- B.1: Linear grading + transitory regions (deterministic only) ----------
    print("  Running B.1 (linear grading + transitory)")
    cfg_b1 = replace(base_cfg, grade_nm=base_cfg.grade_nm, sys_mode_n="linear", sys_params_n={"a": 0.4})
    res_b1 = run_staged_training_with_snapshots(cfg_b1, target,
                                               sys_mode_n="linear", sys_params_n={"a": 0.4},
                                               rnd_gamma=0.0, rnd_mu=0.0,
                                               pre_iters=pre_iters, post_iters=post_iters,
                                               snapshot_iters=snapshot_iters)
    _plot_snapshot_grid(res_b1, "Figure_B1.png", "Figure B.1 — Linear gradient shift (snapshots & loss)", snapshot_iters)

    # ---------- B.2: Sinusoidal height-dependent (deterministic only) ----------
    print("  Running B.2 (sinusoidal height-dependent)")
    cfg_b2 = replace(base_cfg, sys_mode_n="sine", sys_params_n={"a": 0.2, "f": 3.0})
    res_b2 = run_staged_training_with_snapshots(cfg_b2, target,
                                               sys_mode_n="sine", sys_params_n={"a": 0.2, "f": 3.0},
                                               rnd_gamma=0.0, rnd_mu=0.0,
                                               pre_iters=pre_iters, post_iters=post_iters,
                                               snapshot_iters=snapshot_iters)
    _plot_snapshot_grid(res_b2, "Figure_B2.png", "Figure B.2 — Sinusoidal height-dependent shift (snapshots & loss)", snapshot_iters)

    # ---------- B.3: Linear grading + gamma=±1% random ----------
    print("  Running B.3 (linear grading + gamma=±1%)")
    cfg_b3 = replace(base_cfg, sys_mode_n="linear", sys_params_n={"a": 0.4}, rnd_n_gamma=0.01)
    res_b3 = run_staged_training_with_snapshots(cfg_b3, target,
                                               sys_mode_n="linear", sys_params_n={"a": 0.4},
                                               rnd_gamma=0.01, rnd_mu=0.0,
                                               pre_iters=pre_iters, post_iters=post_iters,
                                               snapshot_iters=snapshot_iters)
    _plot_snapshot_grid(res_b3, "Figure_B3.png", "Figure B.3 — Linear gradient + random γ=±1% (snapshots & loss)", snapshot_iters)

    # ---------- B.4: Sinusoidal + γ=±1% ----------
    print("  Running B.4 (sinusoidal + gamma=±1%)")
    cfg_b4 = replace(base_cfg, sys_mode_n="sine", sys_params_n={"a": 0.2, "f": 3.0}, rnd_n_gamma=0.01)
    res_b4 = run_staged_training_with_snapshots(cfg_b4, target,
                                               sys_mode_n="sine", sys_params_n={"a": 0.2, "f": 3.0},
                                               rnd_gamma=0.01, rnd_mu=0.0,
                                               pre_iters=pre_iters, post_iters=post_iters,
                                               snapshot_iters=snapshot_iters)
    _plot_snapshot_grid(res_b4, "Figure_B4.png", "Figure B.4 — Sinusoidal + random γ=±1% (snapshots & loss)", snapshot_iters)

    # ---------- B.5: Linear + γ=±5% ----------
    print("  Running B.5 (linear + gamma=±5%)")
    cfg_b5 = replace(base_cfg, sys_mode_n="linear", sys_params_n={"a": 0.4}, rnd_n_gamma=0.05)
    res_b5 = run_staged_training_with_snapshots(cfg_b5, target,
                                               sys_mode_n="linear", sys_params_n={"a": 0.4},
                                               rnd_gamma=0.05, rnd_mu=0.0,
                                               pre_iters=pre_iters, post_iters=post_iters,
                                               snapshot_iters=snapshot_iters)
    _plot_snapshot_grid(res_b5, "Figure_B5.png", "Figure B.5 — Linear + random γ=±5% (snapshots & loss)", snapshot_iters)

    # ---------- B.6: Sinusoidal + γ=±5% ----------
    print("  Running B.6 (sinusoidal + gamma=±5%)")
    cfg_b6 = replace(base_cfg, sys_mode_n="sine", sys_params_n={"a": 0.2, "f": 3.0}, rnd_n_gamma=0.05)
    res_b6 = run_staged_training_with_snapshots(cfg_b6, target,
                                               sys_mode_n="sine", sys_params_n={"a": 0.2, "f": 3.0},
                                               rnd_gamma=0.05, rnd_mu=0.0,
                                               pre_iters=pre_iters, post_iters=post_iters,
                                               snapshot_iters=snapshot_iters)
    _plot_snapshot_grid(res_b6, "Figure_B6.png", "Figure B.6 — Sinusoida    l + random γ=±5% (snapshots & loss)", snapshot_iters)

    print("Appendix B figures saved to OUTPUT_DIR.")



# -------------------------
# Entry points to call
# -------------------------
def generate_appendices(base_cfg: DesignConfig, fast_mode=False):
    """
    Convenience wrapper: generates Appendix A and B.
    Set fast_mode=True while debugging to reduce iterations.
    """
    run_appendix_A(base_cfg, fast_mode=fast_mode)
    generate_appendix_B(base_cfg, fast_mode=fast_mode)



if __name__ == "__main__":
    print("Running Appendix A & B reproduction…")

    # Create a base configuration
    base_config = DesignConfig(
        num_layers=200,
        n_min=1.6,
        n_max=2.4,
        d_min_nm=20.0,
        d_max_nm=120.0,
        wl_min_nm=400.0,
        wl_max_nm=700.0,
        wl_points=300,
        lr=1e-3,
        iters_stage1=1500,
        grade_nm=0.0,       # Appendix uses *ideal* stacks
        grade_step_nm=4.0,
        sys_mode_n="none",
        sys_mode_d="none",
        rnd_n_mu=0.0,
        rnd_n_gamma=0.0,
        rnd_d_mu_nm=0.0,
        rnd_d_gamma_nm=0.0
    )

    # Call the Appendix generator
    generate_appendices(base_config, fast_mode=False)

    print("All Appendix A & B plots completed.")

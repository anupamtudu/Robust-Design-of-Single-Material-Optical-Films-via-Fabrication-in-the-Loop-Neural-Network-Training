# -*- coding: utf-8 -*-
# Chapter 3 replication:
# - Target variants: band-stop / long-pass / dual-stop
# - Graded interfaces: 10 nm total, 1 nm steps
# - Pruning: MAPE-based (fabrication-aware)

import math, os, json, time
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# -----------------------------
# 0) Global config & utilities
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] {DEVICE}")

# Reproducibility
SEED = 1337
torch.manual_seed(SEED)
np.random.seed(SEED)

@dataclass
class DesignConfig:
    num_layers: int = 100                 # ~100 layers as in ch.3
    n_min: float = 1.6                    # index range for SiN_x-like window
    n_max: float = 2.4
    d_min_nm: float = 20.0                # thickness range [20, 120] nm
    d_max_nm: float = 120.0
    air_n: float = 1.0
    sub_n: float = 1.5
    wl_min_nm: float = 350.0
    wl_max_nm: float = 750.0
    wl_points: int = 400

    # Transition regions (Chapter 3 demo numbers)
    grade_nm: float = 10.0                # total graded-interface width (nm)
    grade_step_nm: float = 1.0            # sublayer size for grading (nm)

    # Systematic drifts (set amplitudes to 0 to disable)
    sys_mode_n: str = "linear"            # ['linear','poly','sine','exp', 'none']
    sys_mode_d: str = "none"              # independent mode for d
    sys_params_n: dict = None             # see make_systematic() for fields
    sys_params_d: dict = None

    # Random noise during Stage-2 (half-range gamma style)
    rnd_n_mu: float = 0.0                 # mean shift on n
    rnd_n_gamma: float = 0.02             # uniform half-range on n (±gamma)
    rnd_d_mu_nm: float = 0.0
    rnd_d_gamma_nm: float = 1.5

    # Training
    lr: float = 1e-3
    iters_stage1: int = 800               # ideal
    iters_stage2: int = 800               # fabrication-in-loop

    # Pruning (MAPE-based, fabrication-aware)
    mc_trials: int = 1000
    pruning_mape_tol_pct: float = 1.0     # stop if best-merge MAPE > this (%)
    pruning_target_layers: int = 0        # 0 → no explicit target; else reduce to <= this
    max_merge_passes: int = 5             # how many merge sweeps
    quick_retune_steps: int = 200         # small retune after pruning

    # Target spectrum kind
    target_mode: str = "band_stop"        # one of ["band_stop","long_pass","dual_stop"]
    target_params: dict = None            # see build_target() for keys

cfg = DesignConfig(
    sys_params_n={"a": 0.20, "p": 1.0, "f": 1.0, "phi": 0.0, "b": 2.0},
    sys_params_d={"a": 0.0,  "p": 1.0, "f": 1.0, "phi": 0.0, "b": 2.0},

    # Choose ONE target (edit these lines to switch targets):
    target_mode="band_stop",
    target_params={"lo_nm": 530.0, "hi_nm": 570.0},


    # target_mode="long_pass",
    # target_params={"cut_nm": 550.0},
    
    # target_mode="dual_stop",
    # target_params={"a0_nm": 450.0, "a1_nm": 500.0, "b0_nm": 600.0, "b1_nm": 650.0},
)

# wavelength grid (meters)
WAVELENGTHS_NM = torch.linspace(cfg.wl_min_nm, cfg.wl_max_nm, cfg.wl_points, device=DEVICE)
WAVELENGTHS_M = WAVELENGTHS_NM * 1e-9

# -----------------------------
# Target builder (chapter variants)
# -----------------------------
def build_target(wavelengths_nm: torch.Tensor, mode: str, params: dict) -> torch.Tensor:
    T = torch.ones_like(wavelengths_nm, device=DEVICE)
    p = params or {}
    if mode == "band_stop":
        lo, hi = float(p.get("lo_nm", 530.0)), float(p.get("hi_nm", 570.0))
        T[(wavelengths_nm > lo) & (wavelengths_nm < hi)] = 0.0
    elif mode == "long_pass":
        cut = float(p.get("cut_nm", 550.0))
        T[wavelengths_nm < cut] = 0.0
    elif mode == "dual_stop":
        a0, a1 = float(p.get("a0_nm", 450.0)), float(p.get("a1_nm", 500.0))
        b0, b1 = float(p.get("b0_nm", 600.0)), float(p.get("b1_nm", 650.0))
        T[(wavelengths_nm > a0) & (wavelengths_nm < a1)] = 0.0
        T[(wavelengths_nm > b0) & (wavelengths_nm < b1)] = 0.0
    else:
        raise ValueError(f"Unknown target_mode: {mode}")
    return T

TARGET = build_target(WAVELENGTHS_NM, cfg.target_mode, cfg.target_params)

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
    for i in range(L):
        Mtot = Mtot @ M[:, i, :, :]

    n0 = n_c[:, 0]
    ns = n_c[:, -1]
    m00, m01 = Mtot[:, 0, 0], Mtot[:, 0, 1]
    m10, m11 = Mtot[:, 1, 0], Mtot[:, 1, 1]

    t = (2 * n0) / (m00 + m01 * ns + n0 * m10 + n0 * m11 * ns)
    T = torch.abs(t) ** 2 * (torch.real(ns) / torch.real(n0))
    return T

# -----------------------------
# 2) Graded interfaces builder
# -----------------------------
def build_graded_stack(n_core, d_core_nm, air_n, sub_n, grade_nm, step_nm):
    """
    Differentiable graded-interface builder.
    - Constant number of graded steps per interface, so shape stays stable.
    - Includes explicit 0-thickness boundary layers (air/substrate).
    Returns:
        n_full: (L_total,) tensor
        d_full_m: (L_total,) meters
    """
    # all scalars as tensors on the right device/dtypes
    air = torch.as_tensor(air_n, device=n_core.device, dtype=n_core.dtype)
    sub = torch.as_tensor(sub_n, device=n_core.device, dtype=n_core.dtype)

    # how many graded sublayers per interface
    steps = int(math.ceil(float(grade_nm) / float(step_nm))) if grade_nm > 0.0 else 0

    if steps > 0:
        # interpolation fractions in (0,1): exclude endpoints
        s_fracs = torch.arange(1, steps + 1, device=n_core.device, dtype=n_core.dtype) / (steps + 1)
        grade_th_nm = torch.full((steps,), grade_nm / steps, device=d_core_nm.device, dtype=d_core_nm.dtype)
    else:
        s_fracs = torch.empty(0, device=n_core.device, dtype=n_core.dtype)
        grade_th_nm = torch.empty(0, device=d_core_nm.device, dtype=d_core_nm.dtype)

    n_chunks = []
    d_chunks_nm = []

    # prepend boundary air (0 nm)
    n_chunks.append(air.view(1))
    d_chunks_nm.append(torch.zeros(1, device=d_core_nm.device, dtype=d_core_nm.dtype))

    # air -> first layer grading
    if steps > 0:
        nL = air.view(1, 1)                    # (1,1)
        nR = n_core[0].view(1, 1)              # (1,1)
        n_g = nL + (nR - nL) * s_fracs.view(-1, 1)   # (steps,1)
        n_chunks.append(n_g.view(-1))
        d_chunks_nm.append(grade_th_nm.clone())

    # iterate over core layers
    L = n_core.shape[0]
    for i in range(L):
        # core layer
        n_chunks.append(n_core[i].view(1))
        d_chunks_nm.append(d_core_nm[i].view(1))

        # grade to next (or substrate)
        if steps > 0:
            nL = n_core[i].view(1, 1)
            nR = (sub if i == L - 1 else n_core[i + 1]).view(1, 1)
            n_g = nL + (nR - nL) * s_fracs.view(-1, 1)   # (steps,1)
            n_chunks.append(n_g.view(-1))
            d_chunks_nm.append(grade_th_nm.clone())

    # append boundary substrate (0 nm)
    n_chunks.append(sub.view(1))
    d_chunks_nm.append(torch.zeros(1, device=d_core_nm.device, dtype=d_core_nm.dtype))

    # concatenate
    n_full = torch.cat(n_chunks, dim=0)                         # (L_total,)
    d_full_nm = torch.cat(d_chunks_nm, dim=0)                   # (L_total,)
    d_full_m = torch.clamp(d_full_nm * 1e-9, min=1e-12)         # keep strictly positive

    return n_full, d_full_m

# -----------------------------
# 3) Systematic drifts & random noise
# -----------------------------
def make_systematic(vec, mode, params):
    """Apply deterministic depth-dependent drift to a 1D vector (n or d)."""
    if mode is None or mode == "none":
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
        # normalized exponential from 0 to 1
        num = torch.exp(b * z) - 1.0
        den = math.e ** b - 1.0
        drift = a * (num / den)
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
# 4) Model: two-head FCNN (n & d)
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
        n_hat = self.out_n(h) * (self.n_max - self.n_min) + self.n_min
        d_hat_nm = self.out_d(h) * (self.d_max - self.d_min) + self.d_min
        return n_hat, d_hat_nm

# -----------------------------
# 5) Loss, evaluation helpers
# -----------------------------
mse = nn.MSELoss()

def forward_spectrum(n_core, d_core_nm, with_grading=True,
                     sys_modes=("none","none"), sys_params=(None,None),
                     rnd_params=None):
    """
    Build full stack (with grading), apply deterministic drifts and random noise,
    and compute T(λ).
    rnd_params: dict with keys {'n_mu','n_gamma','d_mu_nm','d_gamma_nm'} for per-call randomization
    """
    n_det = make_systematic(n_core, sys_modes[0], sys_params[0] or {})
    d_det_nm = make_systematic(d_core_nm, sys_modes[1], sys_params[1] or {})

    if rnd_params is not None:
        n_det = add_random_uniform(n_det, rnd_params.get("n_mu", 0.0), rnd_params.get("n_gamma", 0.0))
        d_det_nm = add_random_uniform(d_det_nm, rnd_params.get("d_mu_nm", 0.0), rnd_params.get("d_gamma_nm", 0.0))

    if with_grading and cfg.grade_nm > 0.0:
        n_full, d_full_m = build_graded_stack(n_det, d_det_nm, cfg.air_n, cfg.sub_n, cfg.grade_nm, cfg.grade_step_nm)
    else:
        # no grading; just the core stack between air and substrate
        n_full = torch.cat([
            torch.tensor([cfg.air_n], device=DEVICE),
            n_det,
            torch.tensor([cfg.sub_n], device=DEVICE)
        ])
        d_full_m = torch.cat([
            torch.tensor([0.0], device=DEVICE),
            d_det_nm * 1e-9,
            torch.tensor([0.0], device=DEVICE)
        ])

    return tmm_normal(n_full, d_full_m, WAVELENGTHS_M)

def mape_percent(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-6) -> float:
    """Return MAPE (%) as a Python float."""
    num = torch.abs(y_pred - y_true)
    den = torch.clamp(torch.abs(y_true), min=eps)
    return float((100.0 * (num / den)).mean().item())

@torch.no_grad()
def prune_mape(
    n_core: torch.Tensor,
    d_core_nm: torch.Tensor,
    target_T: torch.Tensor,
    *,
    mape_tol_pct: float,
    target_layers: int,
    max_passes: int,
    fabrication_aware: bool = True
):
    """
    Greedy adjacent-merge with minimal MAPE increase against current reference spectrum.
    If fabrication_aware=True, spectra include grading + systematic drifts (no random noise).
    """
    def spec(nc, dc_nm):
        return forward_spectrum(
            nc, dc_nm,
            with_grading=fabrication_aware,
            sys_modes=(cfg.sys_mode_n, cfg.sys_mode_d) if fabrication_aware else ("none","none"),
            sys_params=(cfg.sys_params_n, cfg.sys_params_d) if fabrication_aware else (None, None),
            rnd_params=None
        )

    # Use fabrication-aware spectrum as local reference
    T_ref = spec(n_core, d_core_nm)

    n = n_core.clone()
    d = d_core_nm.clone()
    passes = 0

    while passes < max_passes:
        passes += 1
        if n.numel() <= max(2, target_layers if target_layers > 0 else 2):
            break

        best = None  # (mape, i, new_n, new_d)
        # test all adjacent merges, pick lowest MAPE vs current reference
        for i in range(n.numel() - 1):
            tot = d[i] + d[i+1]
            n_eff = 0.5 * (n[i] + n[i+1]) if tot <= 1e-9 else (n[i]*d[i] + n[i+1]*d[i+1]) / tot
            cand_n = torch.cat([n[:i], n_eff.view(1), n[i+2:]])
            cand_d = torch.cat([d[:i], tot.view(1), d[i+2:]])
            T_cand = spec(cand_n, cand_d)
            mape = mape_percent(T_cand, T_ref)
            if (best is None) or (mape < best[0]):
                best = (mape, i, cand_n, cand_d)

        if best is None:
            break

        best_mape, best_i, best_n, best_d = best
        if best_mape > mape_tol_pct:
            # merging would hurt spectrum more than allowed
            break

        # accept best merge
        n, d = best_n, best_d

        # honor explicit layer target if set
        if target_layers > 0 and n.numel() <= target_layers:
            break

        # update reference spectrum around the accepted design
        T_ref = spec(n, d)

    return n.to(DEVICE), d.to(DEVICE)

# (Optional) Keep the old tolerance-based pruning for reference; not used now.
def greedy_prune(n_core, d_core_nm, tol_n, tol_d_nm, max_passes=3):
    n = n_core.clone()
    d = d_core_nm.clone()
    for _ in range(max_passes):
        if n.numel() <= 2:
            break
        new_n = []
        new_d = []
        i = 0
        while i < n.numel():
            if i < n.numel() - 1 and abs(n[i] - n[i+1]) < tol_n and abs(d[i] - d[i+1]) < tol_d_nm:
                tot = d[i] + d[i+1]
                n_eff = 0.5 * (n[i] + n[i+1]) if tot <= 1e-9 else (n[i]*d[i] + n[i+1]*d[i+1]) / tot
                new_n.append(n_eff)
                new_d.append(tot)
                i += 2
            else:
                new_n.append(n[i])
                new_d.append(d[i])
                i += 1
        n = torch.stack(new_n)
        d = torch.stack(new_d)
    return n.to(DEVICE), d.to(DEVICE)

# -----------------------------
# 6) Training
# -----------------------------
def train_ch3():
    seed_vec = torch.randn(cfg.num_layers, device=DEVICE)
    model = OnlineOptimizer(
        seed_size=cfg.num_layers, num_layers=cfg.num_layers,
        n_min=cfg.n_min, n_max=cfg.n_max, d_min_nm=cfg.d_min_nm, d_max_nm=cfg.d_max_nm
    ).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=cfg.lr)

    hist = {"stage1": [], "stage2": [], "stage2_rand": []}

    # Stage 1: Ideal (abrupt interfaces, no drifts/noise)
    print("\n[Stage 1] Ideal optimization")
    for it in range(1, cfg.iters_stage1 + 1):
        opt.zero_grad()
        n_hat, d_hat_nm = model(seed_vec)
        T = forward_spectrum(n_hat, d_hat_nm,
                             with_grading=False,
                             sys_modes=("none","none"), sys_params=(None,None),
                             rnd_params=None)
        loss = mse(T, TARGET)
        loss.backward()
        opt.step()
        hist["stage1"].append(float(loss.item()))
        if it % 100 == 0:
            print(f"  it {it:4d} | L = {loss.item():.6f}")

    # Capture ideal design
    with torch.no_grad():
        n_ideal, d_ideal_nm = model(seed_vec)
        T_ideal = forward_spectrum(n_ideal, d_ideal_nm, with_grading=False)

    # Stage 2: Fabrication-in-the-loop (deterministic drifts + graded interfaces + per-iter random noise)
    print("\n[Stage 2] Fabrication-in-loop with systematic + random perturbations + graded interfaces")
    for it in range(1, cfg.iters_stage2 + 1):
        opt.zero_grad()
        n_hat, d_hat_nm = model(seed_vec)
        T = forward_spectrum(
            n_hat, d_hat_nm,
            with_grading=True,
            sys_modes=(cfg.sys_mode_n, cfg.sys_mode_d),
            sys_params=(cfg.sys_params_n, cfg.sys_params_d),
            rnd_params=dict(n_mu=cdf(cfg.rnd_n_mu), n_gamma=cdf(cfg.rnd_n_gamma),
                            d_mu_nm=cdf(cfg.rnd_d_mu_nm), d_gamma_nm=cdf(cfg.rnd_d_gamma_nm))
        )
        loss = mse(T, TARGET)
        loss.backward()
        opt.step()
        hist["stage2"].append(float(loss.item()))
        if it % 100 == 0:
            print(f"  it {it:4d} | L = {loss.item():.6f}")

    # Final designs
    with torch.no_grad():
        n_final, d_final_nm = model(seed_vec)
        T_ideal_end = forward_spectrum(n_final, d_final_nm, with_grading=False,
                                       sys_modes=("none","none"), sys_params=(None,None))
        T_fab = forward_spectrum(n_final, d_final_nm, with_grading=True,
                                 sys_modes=(cfg.sys_mode_n, cfg.sys_mode_d),
                                 sys_params=(cfg.sys_params_n, cfg.sys_params_d),
                                 rnd_params=None)

    # Monte-Carlo robustness (random errors only, on top of deterministic + grading)
    print("\n[Eval] Monte-Carlo robustness…")
    with torch.no_grad():
        mc_losses = []
        BATCH = 50  # micro-batch to avoid OOM
        rem = 1000 if cfg.iters_stage2 >= 1 else 0  # use cfg.mc_trials if you want; fixed to 1000 here
        # Use cfg.mc_trials from config if you prefer:
        rem = cfg.mc_trials
        while rem > 0:
            b = min(BATCH, rem)
            for _ in range(b):
                T_mc = forward_spectrum(
                    n_final, d_final_nm, with_grading=True,
                    sys_modes=(cfg.sys_mode_n, cfg.sys_mode_d),
                    sys_params=(cfg.sys_params_n, cfg.sys_params_d),
                    rnd_params=dict(n_mu=cfg.rnd_n_mu, n_gamma=cfg.rnd_n_gamma,
                                    d_mu_nm=cfg.rnd_d_mu_nm, d_gamma_nm=cfg.rnd_d_gamma_nm)
                )
                mc_losses.append(float(mse(T_mc, TARGET).item()))
            rem -= b
        mc_mean = float(np.mean(mc_losses))
        mc_std = float(np.std(mc_losses))
        print(f"  MC {cfg.mc_trials} trials | mean MSE={mc_mean:.6f} ± {mc_std:.6f}")

    # Pruning: MAPE-based (fabrication-aware) + quick retune
    print("\n[Pruning] MAPE-based, fabrication-aware layer reduction (Chapter 3 style)")
    with torch.no_grad():
        n_pruned, d_pruned_nm = prune_mape(
            n_final, d_final_nm,
            target_T=TARGET,
            mape_tol_pct=cfg.pruning_mape_tol_pct,
            target_layers=cfg.pruning_target_layers,
            max_passes=cfg.max_merge_passes,
            fabrication_aware=True
        )
        print(f"  Layers: {cfg.num_layers} -> {n_pruned.numel()} (MAPE ≤ {cfg.pruning_mape_tol_pct:.2f}%)")

    # Quick retune after pruning (keep pruned count fixed; nudge within bounds)
    tweak = nn.Parameter(torch.zeros(n_pruned.numel() * 2, device=DEVICE))
    tw_opt = optim.Adam([tweak], lr=5e-3)
    for it in range(1, cfg.quick_retune_steps + 1):
        tw_opt.zero_grad()
        # small deltas in [-0.02, +0.02] for n, and [-2,+2] nm for d via tanh
        dn = 0.02 * torch.tanh(tweak[:n_pruned.numel()])
        dd_nm = 2.0 * torch.tanh(tweak[n_pruned.numel():])
        n_t = torch.clamp(n_pruned + dn, min=cfg.n_min, max=cfg.n_max)
        d_t_nm = torch.clamp(d_pruned_nm + dd_nm, min=cfg.d_min_nm, max=cfg.d_max_nm)
        Tt = forward_spectrum(
            n_t, d_t_nm, with_grading=True,
            sys_modes=(cfg.sys_mode_n, cfg.sys_mode_d),
            sys_params=(cfg.sys_params_n, cfg.sys_params_d),
            rnd_params=None
        )
        L = mse(Tt, TARGET)
        L.backward()
        tw_opt.step()
        if it % 50 == 0:
            print(f"  retune {it:4d} | L = {L.item():.6f}")

    with torch.no_grad():
        n_final_pr, d_final_pr_nm = torch.clamp(n_pruned + 0.02*torch.tanh(tweak[:n_pruned.numel()]),
                                                cfg.n_min, cfg.n_max), \
                                    torch.clamp(d_pruned_nm + 2.0*torch.tanh(tweak[n_pruned.numel():]),
                                                cfg.d_min_nm, cfg.d_max_nm)
        T_pruned = forward_spectrum(
            n_final_pr, d_final_pr_nm, with_grading=True,
            sys_modes=(cfg.sys_mode_n, cfg.sys_mode_d),
            sys_params=(cfg.sys_params_n, cfg.sys_params_d),
            rnd_params=None
        )

    results = dict(
        n_ideal=n_ideal.detach().cpu().numpy().tolist(),
        d_ideal_nm=d_ideal_nm.detach().cpu().numpy().tolist(),
        n_final=n_final.detach().cpu().numpy().tolist(),
        d_final_nm=d_final_nm.detach().cpu().numpy().tolist(),
        n_pruned=n_final_pr.detach().cpu().numpy().tolist(),
        d_pruned_nm=d_final_pr_nm.detach().cpu().numpy().tolist(),
        mc_mean=mc_mean, mc_std=mc_std
    )

    # -----------------------------
    # Plots (Chapter-3 style)
    # -----------------------------
    wl = WAVELENGTHS_NM.detach().cpu().numpy()
    T_tgt = TARGET.detach().cpu().numpy()
    T_id0 = T_ideal.detach().cpu().numpy()
    T_id1 = T_ideal_end.detach().cpu().numpy()
    T_fb = T_fab.detach().cpu().numpy()
    T_pr = T_pruned.detach().cpu().numpy()

    os.makedirs("ch3_out2_NEW", exist_ok=True)

    # Fig A: Stage-1 convergence
    plt.figure(figsize=(7,4))
    plt.plot(hist["stage1"])
    plt.xlabel("Iteration")
    plt.ylabel("MSE")
    plt.title("Stage 1 (Ideal) Convergence")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("ch3_out2_NEW/stage1_convergence.png", dpi=160)

    # Fig B: Stage-2 convergence
    plt.figure(figsize=(7,4))
    plt.plot(hist["stage2"])
    plt.xlabel("Iteration")
    plt.ylabel("MSE")
    plt.title("Stage 2 (Fabrication-in-loop) Convergence")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("ch3_out2_NEW/stage2_convergence.png", dpi=160)

    # Fig C: Spectra (target, ideal-before, ideal-after, after-fabrication, pruned)
    plt.figure(figsize=(10,6))
    plt.plot(wl, T_tgt, 'k--', label="Target", alpha=0.7)
    plt.plot(wl, T_id0, 'b:',  label="Ideal (pre-train snapshot)", alpha=0.8)
    plt.plot(wl, T_id1, 'b-',  label="Ideal (final) ", alpha=0.9)
    plt.plot(wl, T_fb,  'g-',  label="With Fabrication (sys+graded)", linewidth=2.0)
    plt.plot(wl, T_pr,  'm-',  label="Pruned + retuned", linewidth=1.8)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Transmittance")
    plt.title("Spectral responses (Chapter 3 storyline)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xlim(400, 700)
    plt.tight_layout()
    plt.savefig("ch3_out2_NEW/spectra_storyline.png", dpi=180)

    # Fig D: Monte-Carlo loss histogram
    plt.figure(figsize=(7,5))
    plt.hist(mc_losses, bins=40, edgecolor='k', alpha=0.75)
    plt.xlabel("MSE vs Target (MC, sys+graded+random)")
    plt.ylabel("Count")
    plt.title(f"Robustness over {cfg.mc_trials} trials (mean={mc_mean:.4f}, std={mc_std:.4f})")
    plt.tight_layout()
    plt.savefig("ch3_out2_NEW/mc_hist.png", dpi=160)

    # Save JSON summary
    with open("ch3_out2_NEW/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n[Done] Outputs saved in ./ch3_out2_NEW")
    print("  - stage1_convergence.png")
    print("  - stage2_convergence.png")
    print("  - spectra_storyline.png")
    print("  - mc_hist.png")
    print("  - results.json")
    return model, results

# Helper: robust conversion if a tensor slips in
def cdf(x):
    return float(x) if isinstance(x, (int, float)) else (float(x.item()) if torch.is_tensor(x) else x)

if __name__ == "__main__":
    _model, _res = train_ch3()

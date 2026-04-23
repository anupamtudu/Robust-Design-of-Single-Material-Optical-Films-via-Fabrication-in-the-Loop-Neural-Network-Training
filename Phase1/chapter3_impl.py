import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Configuration & Constants ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {DEVICE}")

NUM_LAYERS = 50          # Number of active layers to optimize
SEED_SIZE = NUM_LAYERS   # Input size for the NN (random seed vector)
LAYER_THICKNESS = 30e-9  # Fixed 30nm thickness for this simplified demo

# Wavelength range for simulation (visible spectrum: 400nm - 700nm)
WAVELENGTHS = torch.linspace(400, 700, 300).to(DEVICE)

# Define Targets: Simple Bandstop Filter (reflect light between 530nm and 570nm)
TARGET_SPECTRUM = torch.ones_like(WAVELENGTHS)
TARGET_SPECTRUM[(WAVELENGTHS > 530) & (WAVELENGTHS < 570)] = 0.0

# Fixed material properties (Refractive Indices)
N_AIR = torch.tensor([1.0]).to(DEVICE)
N_SUBSTRATE = torch.tensor([1.5]).to(DEVICE)

# --- 2. Differentiable Physics Solver (Transfer Matrix Method) ---
def differentiable_tmm_normal(n_layers, d_layers, wavelengths):
    """
    PyTorch implementation of TMM for normal incidence.
    Supports backpropagation through refractive indices 'n_layers'.
    """
    num_wavelengths = wavelengths.shape[0]
    num_layers = n_layers.shape[0]

    # Expand inputs for vectorized calculation across all wavelengths
    n_layers_c = n_layers.unsqueeze(0).expand(num_wavelengths, -1).cfloat()
    d_layers_c = d_layers.unsqueeze(0).expand(num_wavelengths, -1).cfloat()
    wl_c = wavelengths.unsqueeze(1).cfloat()

    # Wave vector k = 2 * pi * n / lambda
    k = 2 * np.pi * n_layers_c / wl_c
    # Phase shift phi = k * d
    phi = k * d_layers_c

    # Characteristic Matrix for each layer
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)
    
    # Stack into (Num_WL, Num_Layers, 2, 2) transfer matrices
    M = torch.stack([
        torch.stack([cos_phi, (-1j / n_layers_c) * sin_phi], dim=-1),
        torch.stack([-1j * n_layers_c * sin_phi, cos_phi], dim=-1)
    ], dim=-2)

    # Multiply matrices: M_total = M_0 * M_1 * ... * M_N
    M_total = torch.eye(2, dtype=torch.cfloat, device=DEVICE).unsqueeze(0).repeat(num_wavelengths, 1, 1)
    for i in range(num_layers):
         M_total = torch.matmul(M_total, M[:, i, :, :])

    # Extract transmission coefficient t
    # For normal incidence: t = 2*n0 / (M00 + M01*ns + n0*M10 + n0*M11*ns)
    n0 = n_layers_c[:, 0]   # Air (first layer)
    ns = n_layers_c[:, -1]  # Substrate (last layer)
    m00, m01 = M_total[:, 0, 0], M_total[:, 0, 1]
    m10, m11 = M_total[:, 1, 0], M_total[:, 1, 1]

    t = (2 * n0) / (m00 + m01 * ns + n0 * m10 + n0 * m11 * ns)
    
    # Transmittance T = |t|^2 * (Re(ns) / Re(n0))
    T = torch.abs(t)**2 * (torch.real(ns) / torch.real(n0))
    return T

# --- 3. Neural Network (Online Optimizer) ---
class OnlineOptimizer(nn.Module):
    def __init__(self, num_out, seed_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seed_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_out),
            nn.Sigmoid() # Bounds output to [0, 1]
        )
    def forward(self, seed):
        # Scale [0,1] output to refractive index range [1.6, 2.4] (typical for SiNx)
        return self.net(seed) * (2.4 - 1.6) + 1.6

# --- 4. Simulation of Fabrication Errors ---
def apply_systematic_error(n_layers, error_type='linear'):
    """Applies deterministic errors to layer parameters before solver."""
    if error_type == 'linear':
        # Simulate linear drift: refractive index increases towards top layers
        drift = torch.linspace(0, 0.2, steps=n_layers.shape[0], device=DEVICE)
        return n_layers + drift
    return n_layers

# --- 5. Main Training Loop ---
# Initialize fixed seed, model, and optimizer
FIXED_SEED = torch.randn(SEED_SIZE).to(DEVICE)
model = OnlineOptimizer(NUM_LAYERS, SEED_SIZE).to(DEVICE)
opt = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# Prepare thickness tensor (0 thickness for Air/Substrate boundaries)
d_active = torch.full((NUM_LAYERS,), LAYER_THICKNESS, device=DEVICE)
d_stack = torch.cat([torch.tensor([0.0], device=DEVICE), d_active, torch.tensor([0.0], device=DEVICE)])

history = {'ideal_loss': [], 'fab_loss': []}

print("\n--- STAGE 1: Ideal Optimization ---")
for i in range(500):
    opt.zero_grad()
    n_gen = model(FIXED_SEED)
    
    # Assemble full stack: [Air, Layer1...LayerN, Substrate]
    n_stack = torch.cat([N_AIR, n_gen, N_SUBSTRATE])
    
    # Calculate response & loss
    T = differentiable_tmm_normal(n_stack, d_stack, WAVELENGTHS * 1e-9)
    loss = loss_fn(T, TARGET_SPECTRUM)
    loss.backward()
    opt.step()
    
    history['ideal_loss'].append(loss.item())
    if (i+1) % 100 == 0: print(f"Iter {i+1}: Loss = {loss.item():.6f}")

print("\n--- STAGE 2: Fabrication-in-the-Loop Retraining ---")
# We continue training the SAME model, but now we insert errors into the loop.
for i in range(500):
    opt.zero_grad()
    n_gen = model(FIXED_SEED)
    
    # APPLY ERROR: The NN must now learn to pre-compensate for this drift
    n_imperfect = apply_systematic_error(n_gen, error_type='linear')
    
    n_stack = torch.cat([N_AIR, n_imperfect, N_SUBSTRATE])
    T = differentiable_tmm_normal(n_stack, d_stack, WAVELENGTHS * 1e-9)
    loss = loss_fn(T, TARGET_SPECTRUM)
    loss.backward()
    opt.step()
    
    history['fab_loss'].append(loss.item())
    if (i+1) % 100 == 0: print(f"Iter {i+1}: Loss = {loss.item():.6f}")

print("\nOptimization finished. Generating final plots...")

# --- 6. Visualization ---
with torch.no_grad():
    # Get final design from NN
    n_final_ideal = model(FIXED_SEED)
    # Apply error to see how it performs "in reality"
    n_final_fab = apply_systematic_error(n_final_ideal, 'linear')
    
    # Calculate spectra
    stack_ideal = torch.cat([N_AIR, n_final_ideal, N_SUBSTRATE])
    stack_fab = torch.cat([N_AIR, n_final_fab, N_SUBSTRATE])
    T_ideal = differentiable_tmm_normal(stack_ideal, d_stack, WAVELENGTHS*1e-9)
    T_fab = differentiable_tmm_normal(stack_fab, d_stack, WAVELENGTHS*1e-9)

# Plotting
wl_np = WAVELENGTHS.cpu().numpy()
plt.figure(figsize=(10, 6))
plt.plot(wl_np, TARGET_SPECTRUM.cpu().numpy(), 'k--', label='Target', alpha=0.5)
plt.plot(wl_np, T_ideal.cpu().numpy(), 'b:', label='NN Output (Ideal)')
plt.plot(wl_np, T_fab.cpu().numpy(), 'g-', linewidth=2, label='After Fabrication (Simulated Error)')
plt.title("Stage 2 Results: NN Compensating for Systematic Linear Drift")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Transmittance")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("chapter3_results.png")
print("Results saved to 'chapter3_results.png'")
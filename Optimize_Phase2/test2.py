import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt

# --- 1. PHYSICS ENGINE: Vectorized TMM ---

def calculate_art_jax(thicknesses, n_list, wavelengths):
    """
    Computes Transmittance (T) for a multilayer stack using JAX.
    Maintains energy conservation: R + T + A = 1[cite: 50].
    """
    # Incident medium (Air: 1.0) and Substrate (assume Glass/Air: 1.5)
    n_inc = 1.0
    n_sub = 1.5 
    
    def scan_fun(M_total, layer_params):
        n_layer, d_layer = layer_params
        # Compute wave number and phase [cite: 72]
        k = 2 * jnp.pi * n_layer / wavelengths
        phi = k * d_layer
        
        # Transfer Matrix for a single layer (Normal Incidence) [cite: 72]
        # M = [[cos(phi), i*sin(phi)/n], [i*n*sin(phi), cos(phi)]]
        cos_p = jnp.cos(phi)
        sin_p = jnp.sin(phi)
        
        m00 = cos_p
        m01 = 1j * sin_p / n_layer
        m10 = 1j * n_layer * sin_p
        m11 = cos_p
        
        # Batch matrix multiply across all wavelengths
        M_new = jnp.array([[m00, m01], [m10, m11]])
        M_total = jnp.einsum('ij...,jk...->ik...', M_total, M_new)
        return M_total, None

    # Initialize Identity Matrix for all wavelengths (2, 2, len(wavelengths))
    init_M = jnp.eye(2, dtype=jnp.complex64)[:, :, jnp.newaxis] * jnp.ones(len(wavelengths))
    
    # Run the stack simulation [cite: 206]
    final_M, _ = jax.lax.scan(scan_fun, init_M, (n_list, thicknesses))
    
    # Calculate Transmission Coefficient (t) and Transmittance (T) [cite: 48, 74]
    m00, m01, m10, m11 = final_M[0,0], final_M[0,1], final_M[1,0], final_M[1,1]
    denom = (n_inc * m00 + n_inc * n_sub * m01 + m10 + n_sub * m11)
    t = 2 * n_inc / denom
    T = (n_sub / n_inc) * jnp.abs(t)**2
    
    return T

# --- 2. OPTIMIZATION LOGIC ---

def loss_fn(thicknesses, n_list, wavelengths, target_spectrum):
    """Absolute error merit function[cite: 57, 58]."""
    sim_T = calculate_art_jax(thicknesses, n_list, wavelengths)
    mse_loss = jnp.mean(jnp.square(sim_T - target_spectrum))
    
    # Constraints: 1nm to 300nm 
    lower_bound = jnp.sum(jnp.maximum(0.0, 1.0 - thicknesses)**2)
    upper_bound = jnp.sum(jnp.maximum(0.0, thicknesses - 300.0)**2)
    
    return mse_loss + 0.1 * (lower_bound + upper_bound)

@jax.jit
def train_step(thicknesses, n_list, wavelengths, target_T, opt_state):
    loss, grads = jax.value_and_grad(loss_fn)(thicknesses, n_list, wavelengths, target_T)
    updates, opt_state = optimizer.update(grads, opt_state)
    thicknesses = optax.apply_updates(thicknesses, updates)
    return thicknesses, opt_state, loss

# --- 3. EXECUTION ---

# Hyperparameters
max_layers = 60 # [cite: 193]
epochs = 1000   # [cite: 12]
wavelengths = jnp.linspace(380, 1000, 400) # Visible to NIR range [cite: 186]

# Target: 100% Transmittance between 500-700nm (Bandpass) [cite: 53]
target_T = jnp.where((wavelengths > 500) & (wavelengths < 700), 1.0, 0.0)

# Initial Setup: Alternating SiO2 (1.45) and TiO2 (2.4) 
n_list = jnp.array([1.45 if i % 2 == 0 else 2.40 for i in range(max_layers)])
thicknesses = jnp.ones(max_layers) * 80.0 # Initial guess [cite: 190]

optimizer = optax.adam(learning_rate=0.2) # [cite: 15]
opt_state = optimizer.init(thicknesses)

# Optimization Loop
for i in range(epochs):
    thicknesses, opt_state, loss = train_step(thicknesses, n_list, wavelengths, target_T, opt_state)
    if i % 200 == 0:
        print(f"Epoch {i}, Loss: {loss:.6f}")

# Plot Results
final_T = calculate_art_jax(thicknesses, n_list, wavelengths)
plt.plot(wavelengths, target_T, 'k--', label="Target")
plt.plot(wavelengths, final_T, 'r-', label="Optimized Design")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Transmittance")
plt.legend()
plt.title(f"Optimized {max_layers}-Layer Filter")
plt.show()
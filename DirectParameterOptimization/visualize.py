import torch
import numpy as np
import matplotlib.pyplot as plt

# --- 1. SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wavelengths = torch.linspace(300, 2500, 1000)

# --- 2. LOAD DATA FROM FILES ---
# Load optimized thicknesses (23 values)
try:
    final_thicknesses = np.loadtxt('optimized_thicknesses_direct.txt')
    print("Successfully loaded thicknesses.")
except Exception as e:
    print(f"Error loading thicknesses: {e}")

# Load the saved spectrum (1000 points)
try:
    final_T = np.loadtxt('final_spectrum_direct.csv', delimiter=',')
    print("Successfully loaded spectral data.")
except Exception as e:
    print(f"Error loading spectrum: {e}")

# --- 3. TARGET DEFINITION (For Plotting Overlay) ---
target_T = np.where((wavelengths >= 400) & (wavelengths <= 750), 1.0, 0.0)

# --- 4. PLOTTING ---
plt.figure(figsize=(12, 6), dpi=100)

# Plot the Target Window
plt.fill_between(wavelengths, target_T, color='green', alpha=0.1, label='Target Passband (400-750nm)')

# Plot the Result from PINN
plt.plot(wavelengths, final_T, color='crimson', linewidth=2, label='PINN Design Result')

# Formatting for Thesis/Report
plt.title("DP-Optimized 23-Layer Bandpass Filter", fontsize=14, fontweight='bold')
plt.xlabel("Wavelength (nm)", fontsize=12)
plt.ylabel("Transmittance (Normalized)", fontsize=12)
plt.xlim(300, 2500)
plt.ylim(-0.02, 1.05)

# Cut-off indicators
plt.axvline(400, color='black', linestyle='--', alpha=0.4)
plt.axvline(750, color='black', linestyle='--', alpha=0.4)

plt.legend(loc='upper right', frameon=True)
plt.grid(True, which='both', linestyle=':', alpha=0.5)

# Save high-res figure for documentation
plt.savefig('DPO_23_Layer_Result_Plot.png', bbox_inches='tight')
plt.tight_layout()
plt.show()

# --- 5. PRINT SUMMARY STATS ---
vis_mask = (wavelengths >= 400) & (wavelengths <= 750)
vis_rms = np.sqrt(np.mean((final_T[vis_mask] - 1.0)**2))
total_thickness = np.sum(final_thicknesses)

print("-" * 30)
print(f"FINAL DESIGN SUMMARY")
print(f"Visible Region RMS: {vis_rms:.4f}")
print(f"Total Stack Thickness: {total_thickness:.2f} nm")
print("-" * 30)
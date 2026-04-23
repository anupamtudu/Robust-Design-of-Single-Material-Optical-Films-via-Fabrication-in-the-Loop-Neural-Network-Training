import numpy as np

# Load your 1000-point spectrum
# Note: Ensure this matches your exact filename (e.g., 'CHAMPION_GA_spectrum.csv')
T = np.loadtxt('CHAMPION_GUMBEL_spectrum.csv', delimiter=',')
wavelengths = np.linspace(300, 2500, len(T))

# Define your exact zones
uv_mask = (wavelengths >= 300) & (wavelengths < 400)
vis_mask = (wavelengths >= 400) & (wavelengths <= 750)
ir_mask = (wavelengths > 750) & (wavelengths <= 2500)

# Calculate the physical percentages
uv_transmission = np.mean(T[uv_mask]) * 100
vis_transmission = np.mean(T[vis_mask]) * 100
ir_leakage = np.mean(T[ir_mask]) * 100

# Calculate RMS Metrics for the Visible Passband
# 1. Absolute RMS Error (from perfect 1.0 transmission) - Matches your PyTorch logs
vis_rms_absolute = np.sqrt(np.mean((T[vis_mask] - 1.0)**2))

# 2. Ripple RMS (Flatness / Standard Deviation from its own average)
vis_rms_ripple = np.sqrt(np.mean((T[vis_mask] - np.mean(T[vis_mask]))**2)) * 100

print(f"--- CHAMPION FILM PERFORMANCE ---")
print(f"Visual Transparency (Passband):    {vis_transmission:.1f}%")
print(f"Visual Ripple (Flatness RMS):      {vis_rms_ripple:.2f}%")
print(f"Visual Absolute RMS (from 1.0):    {vis_rms_absolute:.4f}")
print(f"Solar Heat Leakage (IR Pass):      {ir_leakage:.1f}%")
print(f"UV Radiation Leakage:              {uv_transmission:.1f}%")
print(f"Total Heat Blocked (IR Rejection): {100 - ir_leakage:.1f}%")
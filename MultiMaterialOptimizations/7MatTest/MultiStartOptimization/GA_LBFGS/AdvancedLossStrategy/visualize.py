import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

# =========================================================
# 1. LOAD THE DATA
# =========================================================
print("Loading 7-Material optimizer output files...")
try:
    # IMPORTANT: Ensure these filenames match your latest 7-material export!
    transmittance = np.loadtxt('CHAMPION_GA_spectrum.csv', delimiter=',')
    wavelengths = np.linspace(300, 2500, len(transmittance))
    
    # Load the physical design
    thicknesses = np.loadtxt('CHAMPION_GA_thicknesses.txt')
    materials = np.loadtxt('CHAMPION_GA_materials.txt', dtype=int)
    
except FileNotFoundError:
    print("Error: Could not find the output files. Ensure the optimizer has finished running.")
    exit()

# =========================================================
# 2. SETUP THE MATPLOTLIB FIGURE
# =========================================================
# Create a large, high-resolution figure with 2 subplots (stacked vertically)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), dpi=150)
fig.subplots_adjust(hspace=0.3)

# =========================================================
# 3. PLOT 1: OPTICAL SPECTRUM (Transmittance)
# =========================================================
ax1.set_title("Filter Optical Performance: Transmittance Spectrum", fontsize=14, fontweight='bold')
ax1.set_xlabel("Wavelength (nm)", fontsize=12)
ax1.set_ylabel("Transmittance (0.0 to 1.0)", fontsize=12)
ax1.set_xlim(300, 2500)
ax1.set_ylim(-0.05, 1.05)
ax1.grid(True, linestyle='--', alpha=0.6)

# Highlight Target Regions
ax1.axvspan(400, 750, color='green', alpha=0.1, label='Target Passband (Vis)')
ax1.axvspan(750, 2500, color='red', alpha=0.1, label='Target Stopband (IR)')

# Plot the actual data
ax1.plot(wavelengths, transmittance, color='black', linewidth=1.5, label='Achieved Transmittance')
ax1.legend(loc='upper right')

# =========================================================
# 4. PLOT 2: PHYSICAL ARCHITECTURE (Refractive Index Profile)
# =========================================================
ax2.set_title("Physical Architecture: Layer-by-Layer Refractive Index", fontsize=14, fontweight='bold')
ax2.set_xlabel("Physical Depth into Filter (nm)", fontsize=12)
ax2.set_ylabel("Refractive Index (n)", fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.6)

# --- THE 7-MATERIAL DICTIONARIES ---
# Approximate real refractive index for visualization height
index_map = {
    0: 1.45,  # SiO2
    1: 2.40,  # TiO2
    2: 1.76,  # Al2O3
    3: 2.00,  # HfO2
    4: 2.10,  # Cr2O3 (Approx average)
    5: 1.05,  # Cu (Proxy for visible spectrum)
    6: 1.38   # MgF2
}

# Distinct colors for 7 materials
color_map = {
    0: '#1f77b4', # Blue (SiO2)
    1: '#ff7f0e', # Orange (TiO2)
    2: '#2ca02c', # Green (Al2O3)
    3: '#9467bd', # Purple (HfO2)
    4: '#d62728', # Red (Cr2O3)
    5: '#8c564b', # Bronze/Brown (Cu)
    6: '#e377c2'  # Pink (MgF2)
}

name_map = {
    0: 'SiO2', 
    1: 'TiO2', 
    2: 'Al2O3', 
    3: 'HfO2', 
    4: 'Cr2O3', 
    5: 'Cu', 
    6: 'MgF2'
}

current_depth = 0.0
x_coords = [0.0]
y_coords = [1.0] # Starts in Air (n=1.0)

# Build the step-plot coordinates
for i in range(len(thicknesses)):
    mat_idx = materials[i]
    n_val = index_map[mat_idx]
    thick = thicknesses[i]
    
    # Draw the vertical step up/down to the new index
    x_coords.append(current_depth)
    y_coords.append(n_val)
    
    # Draw the horizontal line across the thickness of the layer
    current_depth += thick
    x_coords.append(current_depth)
    y_coords.append(n_val)
    
    # Add colored blocks to make it look like physical layers
    rect = patches.Rectangle((current_depth - thick, 1.0), thick, n_val - 1.0, 
                             linewidth=0, facecolor=color_map[mat_idx], alpha=0.3)
    ax2.add_patch(rect)

# Finish the step-plot into the Substrate (Glass n=1.5)
x_coords.append(current_depth)
y_coords.append(1.5)
x_coords.append(current_depth + 200) # Draw a bit of the substrate
y_coords.append(1.5)

# Plot the heavy black step line
ax2.plot(x_coords, y_coords, color='black', linewidth=2)

# Set dynamic X-axis limits based on total thickness
total_thickness = current_depth
ax2.set_xlim(-50, total_thickness + 200)

# Expand Y-axis slightly to fit MgF2 (1.38) to TiO2 (2.40)
ax2.set_ylim(0.9, 2.6)

# DYNAMIC LEGEND: Only plot legend items for materials the AI actually used!
used_materials = np.unique(materials)
custom_lines = [Line2D([0], [0], color=color_map[m], lw=4, alpha=0.5) for m in used_materials]
custom_labels = [name_map[m] for m in used_materials]
ax2.legend(custom_lines, custom_labels, loc='upper right', title="Materials Used")

# Add total thickness text
ax2.text(0.02, 0.05, f'Total Stack Thickness: {total_thickness:.1f} nm', 
         transform=ax2.transAxes, fontsize=12, fontweight='bold', 
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

# =========================================================
# 5. RENDER AND SAVE
# =========================================================
plt.tight_layout()
plt.savefig('Final_7Material_GA_LLOSS_Topology_Report_3.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Visualization complete! Stack contains {len(used_materials)} unique materials.")
print("Saved as 'Final_7Material_GA_LLOSS__Topology_Report.png'")
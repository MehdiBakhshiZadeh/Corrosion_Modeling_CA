# To find a better animation, please use the saved gif file

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
from matplotlib.colors import ListedColormap

# Set the backend to 'TkAgg' for interactive plotting with Tkinter
matplotlib.use('TkAgg')

# Define the domain and initialize parameters with specific initial damage and corroded cells
def initialize_domain(L_x, L_y, protection_layer_thickness):
    C = np.zeros((L_x, L_y))

    # Set initial damage cells for the top crack
    top_crack_indices = [
        (slice(0, 2), slice(95, 105)),
        (slice(3, 5), slice(96, 104)),
        (slice(6, 8), slice(97, 103)),
        (slice(9, 11), slice(99, 101)),
        (slice(12, 14), slice(95, 101)),
        (slice(15, 17), slice(94, 100)),
    ]

    # Set initial damage cells for the bottom crack
    bottom_crack_indices = [
        (slice(198, 200), slice(95, 105)),
        (slice(195, 197), slice(96, 104)),
        (slice(192, 194), slice(97, 103)),
        (slice(189, 191), slice(96, 102)),
        (slice(186, 188), slice(95, 101)),
        (slice(183, 185), slice(94, 100)),
    ]

    for idx in top_crack_indices + bottom_crack_indices:
        C[idx] = -0.75  # Initial damage (black)

    # Add protective layer
    C[:protection_layer_thickness, :] = -1
    C[-protection_layer_thickness:, :] = -1
    C[:, :protection_layer_thickness] = -1
    C[:, -protection_layer_thickness:] = -1

    return C

# Implement Fick's second law for diffusion with different coefficients for each metal
def diffuse(C, D_top, D_bottom, delta_t, delta_x, delta_y):
    C_new = C.copy()
    random_factor = np.random.uniform(0.9, 1.1, size=C.shape)

    # Apply different diffusion coefficients based on the metal type
    for i in range(1, C.shape[0] - 1):
        for j in range(1, C.shape[1] - 1):
            if C[i, j] != -1 and C[i, j] != -0.75:  # Skip protective layer and initial damage
                if i < C.shape[0] // 2:
                    D = D_top
                else:
                    D = D_bottom
                C_new[i, j] += D * delta_t * (
                    (C[i+1, j] - 2 * C[i, j] + C[i-1, j]) / delta_x ** 2 +
                    (C[i, j+1] - 2 * C[i, j] + C[i, j-1]) / delta_y ** 2
                ) * random_factor[i, j]

    return C_new

# Incorporate reaction kinetics with different corrosion rates for different metals
def apply_reaction_kinetics(C, state, reaction_threshold, corrosion_probability_top, corrosion_probability_low, step):
    new_state = state.copy()
    if step < 50:
        return new_state  # No corrosion until step 50

    corroded = (C >= reaction_threshold) & (state == 0)

    # Determine the metal type for each cell
    is_top_metal = np.arange(state.shape[0]).reshape(-1, 1) < state.shape[0] // 2
    is_low_metal = ~is_top_metal

    # Apply corrosion probabilities
    corrosion_probability = np.where(is_top_metal, corrosion_probability_top, corrosion_probability_low)
    probabilistic_corrosion = np.random.rand(*C.shape) < corrosion_probability

    # Only corrode if at least one neighbor is corroded
    for i in range(1, state.shape[0] - 1):
        for j in range(1, state.shape[1] - 1):
            if corroded[i, j] and np.any(state[i-1:i+2, j-1:j+2] == 1):
                if probabilistic_corrosion[i, j]:
                    new_state[i, j] = 1  # New cells become corroded with specified probability

    return new_state

# Simulation function
def simulate_corrosion(L_x, L_y, D_top, D_bottom, time_steps, delta_x, delta_y, delta_t, protection_layer_thickness,
                       reaction_threshold, corrosion_probability_top, corrosion_probability_low, filename):
    C = initialize_domain(L_x, L_y, protection_layer_thickness)
    state = np.zeros_like(C)

    # Set initial corroded cells in the state array
    initial_corroded_indices = [
        (3, L_y // 2),
        (L_x - 4, L_y // 2)
    ]
    for idx in initial_corroded_indices:
        state[idx] = 1  # Initial corroded cells

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))  # Keep the figure size the same
    corroded_counts = []
    top_metal_counts = []
    bottom_metal_counts = []
    protective_layer_counts = []

    def update(t):
        nonlocal C, state
        if t >= 50:
            C = diffuse(C, D_top, D_bottom, delta_t, delta_x, delta_y)
        state = apply_reaction_kinetics(C, state, reaction_threshold, corrosion_probability_top, corrosion_probability_low, t)

        # Display state: corroded cells are marked with 1, initial damage with -0.75, protective layer with -1,
        # and different colors for different metals
        display_state = np.zeros_like(state, dtype=float)
        display_state[(state == 0) & (np.arange(state.shape[0]).reshape(-1, 1) < state.shape[0] // 2)] = -0.75  # Upper metal
        display_state[(state == 0) & (np.arange(state.shape[0]).reshape(-1, 1) >= state.shape[0] // 2)] = -0.25  # Lower metal
        display_state[C == -1] = -1  # Protective layer
        display_state[C == -0.75] = 0.75  # Initial damage
        display_state[state == 1] = 1  # Corroded cells

        ax1.clear()
        im = ax1.imshow(display_state, animated=True, cmap='inferno', vmin=-1, vmax=1)
        ax1.set_title(f'Time Step: {t + 1}')
        ax1.set_xlabel('X Coordinate')
        ax1.set_ylabel('Y Coordinate')

        # Custom legend
        colors = ['yellow', 'gold', 'midnightblue', 'darkviolet', 'black']
        labels = ['Corroded Cells', 'Initial Damage', 'Top Metal', 'Bottom Metal', 'Protective Layer']
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=10) for c in colors]
        ax1.legend(handles, labels, bbox_to_anchor=(1.28, 1.05), loc='upper left', borderaxespad=0.)

        # Count the number of each type of cell
        corroded_count = np.sum(state == 1)
        top_metal_count = np.sum((state == 0) & (np.arange(state.shape[0]).reshape(-1, 1) < state.shape[0] // 2))
        bottom_metal_count = np.sum((state == 0) & (np.arange(state.shape[0]).reshape(-1, 1) >= state.shape[0] // 2))
        protective_layer_count = np.sum(C == -1)

        corroded_counts.append(corroded_count)
        top_metal_counts.append(top_metal_count)
        bottom_metal_counts.append(bottom_metal_count)
        protective_layer_counts.append(protective_layer_count)

        ax2.clear()
        ax2.plot(corroded_counts, color='orange', label='Corroded Cells')
        ax2.plot(top_metal_counts, color='midnightblue', label='Top Metal')
        ax2.plot(bottom_metal_counts, color='darkviolet', label='Bottom Metal')
        ax2.plot(protective_layer_counts, color='black', label='Protective Layer')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Number of Cells')
        ax2.legend()

        return im,

    ani = animation.FuncAnimation(fig, update, frames=time_steps, repeat=False, interval=50, blit=True)

    # Set the colorbar
    cbar = plt.colorbar(ax1.imshow(np.zeros((L_x, L_y)), animated=True, cmap='inferno', vmin=-1, vmax=1), ax=ax1, orientation='vertical', pad=0.1)
    cbar.ax.set_ylabel('Corrosion Status')

    plt.subplots_adjust(left=0.2, right=0.85, top=0.85, bottom=0.2)  # Adjust plot size
    plt.tight_layout()  # Adjust layout to make space for legend

    # Save the animation as a GIF using Pillow
    ani.save(filename, writer='pillow')

    plt.show()
    return ani

# Example usage with specified initial conditions
L_x, L_y = 200, 200  # Dimensions of the plate
D_top = 1.4  # Diffusion coefficient for the top metal
D_bottom = 1.35  # Diffusion coefficient for the bottom metal
time_steps = 500  # Number of time steps to ensure full corrosion
delta_x = delta_y = 1  # Grid spacing
delta_t = 0.2  # Increased time step to ensure full corrosion
protection_layer_thickness = 2  # Thickness of the protective layer
reaction_threshold = 0.1  # Threshold concentration to trigger corrosion

# Corrosion probabilities
corrosion_probability_top = 0.35  # 35% chance of each cell being corroded in the top metal
corrosion_probability_low = 0.33  # 33% chance of each cell being corroded in the low metal

# Call the simulation function and save the animation
filename = 'corrosion_simulation.gif'
ani = simulate_corrosion(L_x, L_y, D_top, D_bottom, time_steps, delta_x, delta_y, delta_t, protection_layer_thickness,
                         reaction_threshold, corrosion_probability_top, corrosion_probability_low, filename)

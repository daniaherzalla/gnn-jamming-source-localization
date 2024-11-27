import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load the pickle file
with open("interpolated_controlled_path_data_1000_new.pkl", "rb") as f:
    spirals_data = []
    try:
        while True:
            spirals_data.append(pickle.load(f))
    except EOFError:
        pass  # End of file reached


# Function to plot a single trajectory for a specified row
def plot_single_trajectory(row_idx):
    spiral_data = spirals_data[row_idx]
    trajectory = spiral_data["node_positions"]
    noise_per_sample = spiral_data["node_noise"]
    target_point = spiral_data["jammer_position"]

    x_points, y_points = zip(*trajectory)

    plt.figure(figsize=(8, 8))
    plt.scatter(*target_point, color='red', s=10, label='Jammer')
    sc = plt.scatter(x_points, y_points, c=noise_per_sample, cmap='viridis', s=1)
    plt.colorbar(sc, label='Noise Level (dBm)')

    plt.title(f"Spiral Trajectory {row_idx + 1} with Noise")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    # plt.xlim(0, 1500)
    # plt.ylim(0, 1500)
    plt.legend()
    plt.grid()
    plt.show()

# Function to plot a single trajectory for a specified row with AoA arrows
def plot_single_trajectory_with_aoa(row_idx):
    spiral_data = spirals_data[row_idx]
    trajectory = spiral_data["node_positions"]
    noise_per_sample = spiral_data["node_noise"]
    target_point = spiral_data["jammer_position"]
    aoa = spiral_data["angle_of_arrival"]  # AoA in radians for each position

    x_points, y_points = zip(*trajectory)

    # Calculate arrow components for AoA
    arrow_dx = np.cos(aoa)
    arrow_dy = np.sin(aoa)

    plt.figure(figsize=(8, 8))
    plt.scatter(*target_point, color='red', s=10, label='Jammer')
    sc = plt.scatter(x_points, y_points, c=noise_per_sample, cmap='viridis', s=1)
    plt.colorbar(sc, label='Noise Level (dBm)')

    # Add AoA arrows
    plt.quiver(x_points, y_points, arrow_dx, arrow_dy, angles='xy', scale_units='xy', scale=10, color='blue', alpha=0.6, label='AoA')

    plt.title(f"Spiral Trajectory {row_idx + 1} with AoA")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.grid()
    plt.show()

# # # Plot individual trajectories for row 55 and row 96
for i in range(30):
    # plot_single_trajectory(i)
    plot_single_trajectory_with_aoa(i)

# Now proceed to the batch plotting of 50 plots per figure
plots_per_figure = 50
total_figures = 20

# Iterate through each spiral in the saved data and plot in batches of 50
for fig_idx in range(total_figures):
    plt.figure(figsize=(24, 24))  # Set a larger figure size to accommodate 50 subplots

    for plot_idx in range(plots_per_figure):
        data_idx = fig_idx * plots_per_figure + plot_idx
        if data_idx >= len(spirals_data):
            break  # Stop if there are no more spirals to plot

        spiral_data = spirals_data[data_idx]

        # Extract trajectory and noise per sample
        trajectory = spiral_data["node_positions"]
        noise_per_sample = spiral_data["node_noise"]
        target_point = spiral_data["jammer_position"]

        # Separate x and y coordinates of the trajectory
        x_points, y_points = zip(*trajectory)

        # Add a subplot for each spiral
        ax = plt.subplot(10, 5, plot_idx + 1)  # 10 rows, 5 columns for 50 subplots
        ax.scatter(*target_point, color='red', s=10, label='Jammer')
        sc = ax.scatter(x_points, y_points, c=noise_per_sample, cmap='viridis', s=1)

        # Set subplot properties
        ax.set_title(f"Spiral {data_idx + 1}", fontsize=10)
        # ax.set_xlim(0, 1500)
        # ax.set_ylim(0, 1500)
        # ax.legend(loc="upper right", fontsize="small")
        ax.grid()

    # Adjust layout for larger subplots
    plt.subplots_adjust(left=0.05, right=0.85, top=0.95, bottom=0.05, wspace=0.3, hspace=0.3)

    # Add a colorbar to each figure
    cbar_ax = plt.gcf().add_axes([0.88, 0.15, 0.02, 0.7])  # Adjusted position of the colorbar
    plt.colorbar(sc, cax=cbar_ax, label='Noise Level (dBm)')

    # Display the figure with 50 plots
    plt.suptitle(f"Figure {fig_idx + 1}: Spiral Trajectories with Noise", fontsize=16)
    plt.show()

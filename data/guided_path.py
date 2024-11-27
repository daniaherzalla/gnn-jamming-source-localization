import json
import math
import random
import time
import numpy as np
import csv
import pygame
import pickle

# Initialize Pygame
pygame.init()

# Screen dimensions
width, height = 1000, 1000
screen = pygame.display.set_mode((width, height))

# Colors
white = (255, 255, 255)
red = (255, 0, 0)
green = (0, 255, 0)

# Fonts
font = pygame.font.Font(None, 36)

# Function to save data to .pkl file
def save_data_as_pkl(data, filename='controlled_path_data_1000sqm.pkl'):
    with open(filename, 'ab') as pkl_file:  # 'ab' mode to append binary data
        pickle.dump(data, pkl_file)


def dbm_to_linear(dbm):
    """Convert dBm to linear scale (milliwatts)."""
    return 10 ** (dbm / 10)


def linear_to_db(linear):
    """Convert linear scale (milliwatts) to dB."""
    return 10 * np.log10(linear)

# Calculate RSSI using simple log-distance path loss model, ignoring antenna direction
def calculate_omni_rssi(distance, P_tx_jammer, G_tx_jammer, path_loss_exponent, shadowing, pl0=32, d0=1):
    # Prevent log of zero if distance is zero by replacing it with a very small positive number
    d = np.where(distance == 0, np.finfo(float).eps, distance)
    # Path loss calculation
    path_loss_db = pl0 + 10 * path_loss_exponent * np.log10(d / d0)

    # Apply shadowing if sigma is not zero
    if shadowing != 0:
        path_loss_db += np.random.normal(0, shadowing, size=d.shape)

    return P_tx_jammer + G_tx_jammer - path_loss_db


# Function to calculate velocity adjustment towards the target
def adjust_velocity(pos, target, vel, coeff):
    dx = target[0] - pos[0]
    dy = target[1] - pos[1]
    dist = math.hypot(dx, dy)
    dx /= dist
    dy /= dist
    vx, vy = vel
    vx += coeff * (dx - vx)
    vy += coeff * (dy - vy)
    return vx, vy

def aoa_error_parameters(noise_level_db, lb_mean=360, ub_mean=0, lb_std=100, ub_std=1, noise_min=-80, noise_max=0):
    """
    Returns the mean and standard deviation of the AoA error based on the noise level in dB.
    The mean and std are linearly interpolated between provided bounds:
    - At noise_level_db = noise_min (default -80 dB), mean = lb_mean, std = lb_std
    - At noise_level_db = noise_max (default -30 dB), mean = ub_mean, std = ub_std

    Parameters:
    - noise_level_db: The noise level in dB at which to calculate the AoA error parameters.
    - lb_mean: The mean at the lowest noise level (default 360).
    - ub_mean: The mean at the highest noise level (default 0).
    - lb_std: The standard deviation at the lowest noise level (default 100).
    - ub_std: The standard deviation at the highest noise level (default 1).
    - noise_min: The lowest noise level in dB for interpolation (default -80 dB).
    - noise_max: The highest noise level in dB for interpolation (default -30 dB).

    Returns:
    - mean: The interpolated mean value.
    - std: The interpolated standard deviation.
    """
    # Handle cases where noise_level_db is outside the bounds
    if noise_level_db <= noise_min:
        return lb_mean, lb_std
    elif noise_level_db >= noise_max:
        return ub_mean, ub_std

    # Calculate the mean (interpolated)
    mean_slope = (ub_mean - lb_mean) / (noise_max - noise_min)  # Change in mean over change in noise level
    mean_intercept = lb_mean - mean_slope * noise_min  # Solve for intercept using one of the points
    mean = mean_slope * noise_level_db + mean_intercept

    # Calculate the std (interpolated)
    std_slope = (ub_std - lb_std) / (noise_max - noise_min)  # Change in std over change in noise level
    std_intercept = lb_std - std_slope * noise_min  # Solve for intercept using one of the points
    std = std_slope * noise_level_db + std_intercept

    return mean, std


def run_simulation():
    # Reset simulation state
    point_a = (random.randint(0, width), random.randint(0, height))
    P_tx_jammer = random.uniform(20, 60)  # dBm
    antenna_gain = random.uniform(0, 5)  # dBi
    path_loss_exponent = random.uniform(2.7, 3.5)
    shadowing = np.random.uniform(2, 6)
    sampling_frequency = 10000 #random.uniform(100, 1000)  # Every second
    noise_level = -100

    # Starting conditions for the drone
    side = random.choice(['top', 'bottom', 'left', 'right'])
    drone_pos = {
        'top': (random.randint(0, width), 0),
        'bottom': (random.randint(0, width), height),
        'left': (0, random.randint(0, height)),
        'right': (width, random.randint(0, height))
    }[side]
    alignment_coefficient = np.random.uniform(0.00015, 0.0025) # (0.00025, 0.015)
    drone_speed = 1 #np.random.uniform(5, 15)
    velocity = (drone_speed, drone_speed)
    speed = math.hypot(*velocity)
    velocity = (velocity[0] / speed, velocity[1] / speed)

    total_angle = 0
    previous_angle = math.atan2(drone_pos[1] - point_a[1], drone_pos[0] - point_a[0])
    D = 10  # Distance threshold
    # last_sample_time = time.time()
    last_sample_time = time.perf_counter()

    noise_values = []
    positions = []
    angles = []
    timestamps = []
    running = True
    # clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                return False  # Indicates to stop all simulations

        # current_time = time.time()
        current_time = time.perf_counter()
        if current_time - last_sample_time >= 1 / sampling_frequency:
            distance_to_jammer = math.hypot(point_a[0] - drone_pos[0], point_a[1] - drone_pos[1])
            rssi = calculate_omni_rssi(distance_to_jammer, P_tx_jammer, antenna_gain, path_loss_exponent, shadowing)

            # Compute linear powers for jammer and normal signals
            N_linear = dbm_to_linear(noise_level)
            jammer_rssi_linear = dbm_to_linear(rssi)

            # Compute SINR in linear scale and convert SINR from linear scale to dB
            noise = jammer_rssi_linear + N_linear
            noise_dB = linear_to_db(noise)

            # Cap jammer RSSI (noise values) based on hardware
            # noise_dB = 0 if noise_dB > 0 else noise_dB
            noise_dB = -80 if noise_dB < -80 else noise_dB

            # Calculate raw AoA
            angle_to_jammer_rad = math.atan2(point_a[1] - drone_pos[1], point_a[0] - drone_pos[0])
            angle_to_jammer_deg = (math.degrees(angle_to_jammer_rad) + 360) % 360

            # Introduce AoA error based on noise and environment
            aoa_err_mean, aoa_err_std = aoa_error_parameters(noise_dB)
            aoa_error = np.random.normal(aoa_err_mean, aoa_err_std)
            adjusted_angle_to_jammer_deg = (angle_to_jammer_deg + aoa_error) % 360

            # Append data to lists
            noise_values.append(noise_dB)
            positions.append(list(drone_pos))
            angles.append(adjusted_angle_to_jammer_deg)
            timestamps.append(current_time)

            last_sample_time = current_time

            # print("Angle to Jammer (degrees):", angle_to_jammer_deg)

        # Update drone's position
        velocity = adjust_velocity(drone_pos, point_a, velocity, alignment_coefficient)
        drone_pos = (drone_pos[0] + velocity[0] * drone_speed, drone_pos[1] + velocity[1] * drone_speed)

        # Check completion conditions
        current_angle = math.atan2(drone_pos[1] - point_a[1], drone_pos[0] - point_a[0])
        angle_difference = current_angle - previous_angle
        if angle_difference < -math.pi:
            angle_difference += 2 * math.pi
        elif angle_difference > math.pi:
            angle_difference -= 2 * math.pi
        total_angle += angle_difference
        previous_angle = current_angle

        # Check conditions
        has_completed_round = abs(total_angle) >= 2 * math.pi
        is_within_distance = math.hypot(drone_pos[0] - point_a[0], drone_pos[1] - point_a[1]) <= D

        # Visualization parameters
        # screen.fill((0, 0, 0))
        # pygame.draw.circle(screen, red, point_a, 5)
        # pygame.draw.circle(screen, white, (int(drone_pos[0]), int(drone_pos[1])), 10)

        # Display status text
        status_text = f"Went around: {has_completed_round} | Distance < {D}: {is_within_distance}"
        text = font.render(status_text, True, green)
        screen.blit(text, (50, height - 40))

        if has_completed_round and is_within_distance:
            if len(positions) > 2:
                # Check RSSI condition
                if sum(rssi > -55 for rssi in noise_values) >= 3:
                    data_dict = {
                        'num_samples': len(positions),
                        'node_positions': positions,
                        'node_noise': noise_values,
                        'angle_of_arrival': angles,
                        'speed': drone_speed,
                        'pl_exp': path_loss_exponent,
                        'sigma': shadowing,
                        'alignment_coefficient': alignment_coefficient,
                        'sampling_frequency': sampling_frequency,
                        'jammer_power': P_tx_jammer,
                        'jammer_position': list(point_a),
                        'jammer_gain': antenna_gain,
                        'dataset': 'dynamic_controlled_path'
                    }
                    save_data_as_pkl(data_dict)
                    return True

        # pygame.display.flip()
        # clock.tick(60)

    return True  # Normal end of a simulation run


# Run 1000 simulations until we get 1000 valid instances
valid_instance_count = 0
while valid_instance_count < 1000:
    if run_simulation():
        print(valid_instance_count)
        valid_instance_count += 1
    else:
        break  # Exit if the simulation signaled to stop


pygame.quit()

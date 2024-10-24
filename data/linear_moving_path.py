import pygame
import math
import random
import time
import numpy as np
import csv

# Initialize Pygame
pygame.init()

# Screen dimensions
width, height = 1000, 1000
screen = pygame.display.set_mode((width, height))

# Colors and fonts
white = (255, 255, 255)
red = (255, 0, 0)
green = (0, 255, 0)
font = pygame.font.Font(None, 36)

# Setup CSV file
csv_file = open('linear_path_data.csv', 'a', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    'timestamps', 'num_samples', 'node_positions', 'node_noise', 'angle_of_arrival', 'speed',
    'pl_exp', 'sigma', 'sampling_frequency',
    'jammer_power', 'jammer_position', 'jammer_gain', 'dataset'
])

def calculate_omni_rssi(distance, P_tx_jammer, antenna_gain, path_loss_exponent, shadowing, pl0=32, d0=1):
    d = np.where(distance == 0, np.finfo(float).eps, distance)
    path_loss_db = pl0 + 10 * path_loss_exponent * np.log10(d / d0)
    # Apply shadowing if sigma is not zero
    if shadowing != 0:
        path_loss_db += np.random.normal(0, shadowing, size=d.shape)
    return P_tx_jammer + antenna_gain - path_loss_db

def dbm_to_linear(dbm):
    """Convert dBm to linear scale (milliwatts)."""
    return 10 ** (dbm / 10)


def linear_to_db(linear):
    """Convert linear scale (milliwatts) to dB."""
    return 10 * np.log10(linear)

def adjust_velocity(pos, target, speed):
    dx, dy = target[0] - pos[0], target[1] - pos[1]
    dist = math.hypot(dx, dy)
    return (dx / dist * speed, dy / dist * speed)  # Scale speed randomly

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
    jammer_pos = (random.randint(0, width), random.randint(0, height))
    P_tx_jammer = random.uniform(20, 60)  # dBm
    antenna_gain = random.uniform(0, 5)  # dBi
    path_loss_exponent = random.uniform(2.7, 3.5)
    shadowing = np.random.uniform(2, 6)
    # speed = random.uniform(5, 20)
    speed = 1
    sampling_frequency = random.uniform(100, 500)  # Every second
    noise_level = -100

    # Define drone's start position and target position randomly on different edges
    sides = ['top', 'bottom', 'left', 'right']
    start_side = random.choice(sides)
    target_sides = [side for side in sides if side != start_side]
    target_side = random.choice(target_sides)

    if start_side == 'top':
        drone_pos = (random.randint(0, width), 0)
    elif start_side == 'bottom':
        drone_pos = (random.randint(0, width), height)
    elif start_side == 'left':
        drone_pos = (0, random.randint(0, height))
    else:
        drone_pos = (width, random.randint(0, height))

    if target_side == 'top':
        target_pos = (random.randint(0, width), 0)
    elif target_side == 'bottom':
        target_pos = (random.randint(0, width), height)
    elif target_side == 'left':
        target_pos = (0, random.randint(0, height))
    else:
        target_pos = (width, random.randint(0, height))

    velocity = adjust_velocity(drone_pos, target_pos, speed)
    last_sample_time = time.time()

    # Lists to store data
    timestamps = []
    positions = []
    noise_values = []
    angles = []

    clock = pygame.time.Clock()
    running = True
    instance_count = 0  # Initialize instance counter

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                return False, instance_count  # Stop all simulations

        # Collect RSSI and position data
        current_time = time.time()
        if current_time - last_sample_time >= 1 / sampling_frequency:
            distance_to_jammer = math.hypot(jammer_pos[0] - drone_pos[0], jammer_pos[1] - drone_pos[1])
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
            angle_to_jammer_rad = math.atan2(jammer_pos[1] - drone_pos[1], jammer_pos[0] - drone_pos[0])
            angle_to_jammer_deg = (math.degrees(angle_to_jammer_rad) + 360) % 360

            # Introduce AoA error based on noise and environment
            aoa_err_mean, aoa_err_std = aoa_error_parameters(noise_dB)
            aoa_error = np.random.normal(aoa_err_mean, aoa_err_std)
            adjusted_angle_to_jammer_deg = (angle_to_jammer_deg + aoa_error) % 360

            # Append data to lists
            positions.append(list(drone_pos))
            noise_values.append(noise_dB)
            angles.append(adjusted_angle_to_jammer_deg)
            timestamps.append(current_time)
            last_sample_time = current_time
            # # Only append data if the drone position is within valid bounds
            # if drone_pos[0] >= 0 and drone_pos[1] >= 0:
            #     positions.append(list(drone_pos))
            #     noise_values.append(noise_dB)
            #     angles.append(adjusted_angle_to_jammer_deg)
            #     timestamps.append(current_time)
            #     last_sample_time = current_time

        # Update drone's position linearly
        drone_pos = (drone_pos[0] + velocity[0], drone_pos[1] + velocity[1])

        # Check if the drone has reached the target position or moved out of bounds
        if not (0 <= drone_pos[0] <= width and 0 <= drone_pos[1] <= height):
            if len(positions) > 2:
                # Save the collected data
                if sum(rssi > -55 for rssi in noise_values) >= 3: # RSSI condition
                    csv_writer.writerow([
                        timestamps,
                        len(positions),
                        positions,
                        noise_values,
                        angles,
                        velocity,
                        path_loss_exponent,
                        shadowing,
                        sampling_frequency,
                        P_tx_jammer,
                        list(jammer_pos),
                        antenna_gain,
                        'dynamic_linear_path'])
                    instance_count += 1  # Increment valid instance count
                break  # End the simulation
            else:
                break

        # Visualization and display update
        screen.fill((0, 0, 0))
        pygame.draw.circle(screen, red, jammer_pos, 10)
        pygame.draw.circle(screen, white, (int(drone_pos[0]), int(drone_pos[1])), 10)
        pygame.display.flip()
        clock.tick(60)

    return True, instance_count  # Reset the simulation for another run

# Run simulations
total_valid_instances = 0
while total_valid_instances < 4000:
    run, count = run_simulation()
    total_valid_instances += count
    if not run:
        break

print(f"Total valid instances: {total_valid_instances}")

csv_file.close()
pygame.quit()

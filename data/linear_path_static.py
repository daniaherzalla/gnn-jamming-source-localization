import math
import random
import numpy as np
import csv
import matplotlib.pyplot as plt

# Setup CSV file
csv_file = open('linear_path_static.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    'num_samples', 'node_positions', 'node_noise', 'angle_of_arrival',
    'pl_exp', 'sigma',
    'jammer_power', 'jammer_position', 'jammer_gain', 'dataset'
])

def dbm_to_linear(dbm):
    return 10 ** (dbm / 10)

def linear_to_db(linear):
    return 10 * np.log10(linear)

def calculate_omni_rssi(distance, P_tx_jammer, antenna_gain, path_loss_exponent, shadowing, pl0=32, d0=1):
    d = np.where(distance == 0, np.finfo(float).eps, distance)
    path_loss_db = pl0 + 10 * path_loss_exponent * np.log10(d / d0)
    if shadowing != 0:
        path_loss_db += np.random.normal(0, shadowing, size=d.shape)
    return P_tx_jammer + antenna_gain - path_loss_db

def generate_points(start_pos, end_pos, num_points):
    x_values = np.linspace(start_pos[0], end_pos[0], num=num_points)
    y_values = np.linspace(start_pos[1], end_pos[1], num=num_points)
    return list(zip(x_values, y_values))

def aoa_error_parameters(noise_level_db, lb_mean=360, ub_mean=0, lb_std=100, ub_std=1, noise_min=-80, noise_max=0):
    if noise_level_db <= noise_min:
        return lb_mean, lb_std
    elif noise_level_db >= noise_max:
        return ub_mean, ub_std
    mean_slope = (ub_mean - lb_mean) / (noise_max - noise_min)
    mean = mean_slope * (noise_level_db - noise_min) + lb_mean
    std_slope = (ub_std - lb_std) / (noise_max - noise_min)
    std = std_slope * (noise_level_db - noise_min) + lb_std
    return mean, std

def plot_instance(positions, noise_values, angles, shadowings, rssi_values, jammer_pos, P_tx_jammer):
    plt.figure(figsize=(10, 10))
    for pos, noise, angle, shadowing, rssi in zip(positions, noise_values, angles, shadowings, rssi_values):
        color = 'red' if noise > -55 else 'blue'
        dx = math.cos(math.radians(angle)) * 20
        dy = math.sin(math.radians(angle)) * 20
        plt.scatter(pos[0], pos[1], color=color, s=100, label='Nodes' if pos == positions[0] else "")
        plt.arrow(pos[0], pos[1], dx, dy, head_width=10, head_length=15, fc='orange', ec='orange')
        plt.text(pos[0], pos[1] + 10, f'RSSI:{rssi:.2f}', color='black', fontsize=8)
        # plt.text(pos[0], pos[1] + 10, f'σ:{shadowing:.2f}\nRSSI:{rssi:.2f}', color='black', fontsize=8)
        # plt.text(pos[0], pos[1] + 10, f'σ:{shadowing:.2f}', color='black', fontsize=8)
    plt.scatter(jammer_pos[0], jammer_pos[1], color='green', s=200, marker='*', label='Jammer')
    plt.title(f'Node Distribution with Noise Levels, AoA, and Jammer Power: {P_tx_jammer:.2f} dBm')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True)
    plt.legend()
    plt.show()

def run_simulation():
    width, height = 1000, 1000
    jammer_pos = (random.randint(0, width), random.randint(0, height))
    P_tx_jammer = random.uniform(20, 60)
    antenna_gain = random.uniform(0, 5)
    path_loss_exponent = random.uniform(2.7, 3.5)
    noise_level = -100

    start_pos = (random.randint(0, width), random.randint(0, height))
    end_pos = (random.randint(0, width), random.randint(0, height))
    num_points = int(np.random.beta(4, 10) * 5000) + 1
    positions = generate_points(start_pos, end_pos, num_points)

    positions_data, noise_values, angles, shadowings, rssi_values = [], [], [], [], []
    for pos in positions:
        shadowing = np.random.uniform(2, 6)
        distance_to_jammer = math.hypot(jammer_pos[0] - pos[0], jammer_pos[1] - pos[1])
        rssi = calculate_omni_rssi(distance_to_jammer, P_tx_jammer, antenna_gain, path_loss_exponent, shadowing)
        noise = dbm_to_linear(rssi) + dbm_to_linear(noise_level)
        noise_dB = linear_to_db(noise)
        noise_dB = max(noise_dB, -80)
        angle_to_jammer_rad = math.atan2(jammer_pos[1] - pos[1], jammer_pos[0] - pos[0])
        angle_to_jammer_deg = (math.degrees(angle_to_jammer_rad) + 360) % 360
        aoa_err_mean, aoa_err_std = aoa_error_parameters(noise_dB)
        aoa_error = np.random.normal(aoa_err_mean, aoa_err_std)
        adjusted_angle_to_jammer_deg = (angle_to_jammer_deg + aoa_error) % 360

        positions_data.append(list(pos))
        noise_values.append(noise_dB)
        angles.append(adjusted_angle_to_jammer_deg)
        shadowings.append(shadowing)
        rssi_values.append(rssi)

    # plot_instance(positions_data, noise_values, angles, shadowings, rssi_values, jammer_pos, P_tx_jammer)

    if sum(noise > -55 for noise in noise_values) >= 3:
        csv_writer.writerow([
            len(positions_data), positions_data, noise_values, angles,
            path_loss_exponent, shadowings, P_tx_jammer,
            list(jammer_pos), antenna_gain, 'dynamic_linear_path'
        ])
        return True
    return False

def generate_instances(target_instances):
    valid_instances = 0
    while valid_instances < target_instances:
        if run_simulation():
            valid_instances += 1
        print(f"Completed instances: {valid_instances}")

generate_instances(5000)  # Demonstrative reduced number
csv_file.close()

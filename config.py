params = {
    'model': 'GATv2',
    'learning_rate': 0.005,
    'weight_decay': 0.0001,
    'batch_size': 16,
    'dropout_rate': 0.6,
    'num_heads': 8,
    'max_epochs': 200,
    'seed': 42,  # opts: 42, 1, 23
    '3d': False,
    'required_features': ['node_positions', 'node_noise'],  # node_positions, polar_coordinates, node_noise, node_rssi
    'additional_features': ['node_states', 'dist_to_centroid', 'sin_azimuth', 'cos_azimuth', 'relative_noise', 'proximity_count', 'clustering_coefficient', 'mean_noise', 'median_noise', 'std_noise', 'range_noise'],  # 'elevation_angle', 'sample_connectivity'
    'coords': 'cartesian',  # opts: 'polar', 'cartesian'
    'num_neighbors': 5,
    'in_channels': 3,  # x, y, z, rssi # r, sin(theta), cos(theta), sin(phi), cos(phi)), rssi
    'num_layers': 4,
    'hidden_channels': 256,
    'out_channels': 64,
    'out_features': 2,  # 3 jammer pos (x, y, z) # 5 (r, sin(theta), cos(theta), sin(phi), cos(phi))
    'edges': 'knn',  # opts: 'knn', 'proximity'
    'norm': 'minmax',  # opts: 'minmax', 'unit_sphere'
    'activation': False,
    'dataset_path': '/home/dania/Downloads/dataset/random/random.csv',  # data/static_swarm_3d.csv # /home/dania/Downloads/dataset/random/random.csv
    'train_path': 'data/train.gzip',
    'val_path': 'data/validation.gzip',
    'test_path': 'data/test.gzip',
    'inference': False,
    'trial_num': 1
}

# CHECK: did you update...
# feats
# in_channels
# out_features
# norm


# 'mean_noise', 'median_noise', 'std_noise', 'range_noise',
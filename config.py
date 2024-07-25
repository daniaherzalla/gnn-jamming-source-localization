params = {
    'model': 'Sage',
    'learning_rate': 0.00023722507100215495,
    'weight_decay': 5.782108151750701e-07,
    'batch_size': 32,
    'dropout_rate': 0.47357262045942095,
    'num_heads': 4,
    'max_epochs': 200,
    'seed': 42,  # opts: 42, 1, 23
    '3d': False,
    'required_features': ['node_positions', 'node_noise'],  # node_positions, polar_coordinates, node_noise, node_rssi
    'additional_features': ['dist_to_centroid', 'sin_azimuth', 'cos_azimuth', 'relative_noise', 'range_noise'],
    'coords': 'cartesian',  # opts: 'polar', 'cartesian'
    'num_neighbors': 20,
    'in_channels': 3,  # x, y, z, rssi # r, sin(theta), cos(theta), sin(phi), cos(phi)), rssi
    'num_layers': 2,
    'hidden_channels': 512,
    'out_channels': 128,
    'out_features': 2,  # 3 jammer pos (x, y, z) # 5 (r, sin(theta), cos(theta), sin(phi), cos(phi))
    'edges': 'knn',  # opts: 'knn', 'proximity'
    'norm': 'minmax',  # opts: 'minmax', 'unit_sphere'
    'activation': False,
    'dataset_path': '/home/dania/gnn-jamming-source-localization/data/triangle.csv',  # data/static_swarm_3d.csv # /home/dania/Downloads/dataset/random/random.csv
    'train_path': 'data/train.gzip',
    'val_path': 'data/validation.gzip',
    'test_path': 'data/test.gzip',
    'inference': False,
    'trial_num': 1,
    'save_data': False,
    'reproduce': False
}

# CHECK: did you update...
# feats
# in_channels
# out_features
# norm


# 'additional_features': ['node_states', 'dist_to_centroid', 'sin_azimuth', 'cos_azimuth', 'relative_noise', 'proximity_count', 'clustering_coefficient', 'mean_noise', 'median_noise', 'std_noise', 'range_noise'],  # 'elevation_angle', 'sample_connectivity'
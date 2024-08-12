params = {
    'model': 'Sage',
    'learning_rate': 0.0005116557930473443,
    'weight_decay': 0.00003128243104068232,
    'batch_size': 16,
    'dropout_rate': 0.2027678869094044,
    'num_heads': 2,
    'max_epochs': 200,
    '3d': False,
    'required_features': ['node_positions', 'node_noise'],  # node_positions, polar_coordinates, node_noise, node_rssi
    'additional_features': [],  # ['relative_noise', 'proximity_count', 'std_noise', 'range_noise'],
    'coords': 'polar',  # opts: 'polar', 'cartesian'
    'num_neighbors': 10,
    'in_channels': 3,  # x, y, z, rssi # r, sin(theta), cos(theta), sin(phi), cos(phi)), rssi
    'num_layers': 4,
    'hidden_channels': 512,
    'out_channels': 32,
    'out_features': 2,  # 3 jammer pos (x, y, z) # 5 (r, sin(theta), cos(theta), sin(phi), cos(phi))
    'edges': 'knn',  # opts: 'knn', 'proximity'
    'norm': 'minmax',  # opts: 'minmax', 'unit_sphere'
    'activation': False,
    'dataset_type': 'rectangle',  # name of experiments_data folder
    'dataset': 'log_distance/urban_area/',  # 'log_distance/urban_area/rectangle/',  # name of experiments_data folder
    'dataset_path': 'data/train_test_data/log_distance/urban_area/combined_urban_area.csv',  # data/static_swarm_3d.csv # /home/dania/Downloads/dataset/random/random.csv
    'inference': False,
    'save_data': False,
    'reproduce': True,  # set False only for dataset study
    'plot_network': False,
    'study': 'coord_system'  # dataset, coord_system, feat_engineering
}

#     'train_path': 'experiments_datasets/datasets/log_distance/urban_area/urban_area/train_dataset.pkl',
#     'val_path': 'experiments_datasets/datasets/log_distance/urban_area/urban_area/validation_dataset.pkl',
#     'test_path': 'experiments_datasets/datasets/log_distance/urban_area/urban_area/test_dataset.pkl',
# 'seed': 100,  # opts: 42, 1, 23

# CHECK: did you update...
# feats
# in_channels
# out_features
# norm


# 'additional_features': ['node_states', 'dist_to_centroid', 'sin_azimuth', 'cos_azimuth', 'relative_noise', 'proximity_count', 'clustering_coefficient', 'mean_noise', 'median_noise', 'std_noise', 'range_noise'],  # 'elevation_angle', 'sample_connectivity'
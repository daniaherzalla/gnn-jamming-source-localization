params = {
    'model': 'Sage',
    'learning_rate': 0.0005116557930473443,
    'weight_decay': 0.00003128243104068232,
    'batch_size': 16,
    'dropout_rate': 0.2027678869094044,
    'num_heads': 2,
    'max_epochs': 200,
    '3d': False,
    'coords': 'cartesian',  # opts: 'polar', 'cartesian'
    'required_features': ['node_positions', 'node_noise'],  # node_positions, polar_coordinates, node_noise, node_rssi
    'additional_features': [],  # ['sin_azimuth', 'cos_azimuth', 'relative_noise', 'proximity_count', 'std_noise', 'range_noise'],  # ['sin_azimuth', 'cos_azimuth', 'relative_noise', 'proximity_count', 'std_noise', 'range_noise'],
    'num_neighbors': 50,
    'in_channels': 4,  # x, y, z, rssi # r, sin(theta), cos(theta), sin(phi), cos(phi)), rssi
    'num_layers': 4,
    'hidden_channels': 512,
    'out_channels': 32,
    'out_features': 2,  # 3 jammer pos (x, y, z) # 5 (r, sin(theta), cos(theta), sin(phi), cos(phi))
    'edges': 'knn',  # opts: 'knn', 'proximity'
    'norm': 'minmax',  # opts: 'minmax', 'unit_sphere'
    'activation': False,
    'test_sets': ['test_dataset.pt', 'circle_test_set.pt', 'triangle_test_set.pt', 'rectangle_test_set.pt', 'random_test_set.pt', 'circle_jammer_outside_region_test_set.pt', 'triangle_jammer_outside_region_test_set.pt', 'rectangle_jammer_outside_region_test_set.pt', 'random_jammer_outside_region_test_set.pt', 'all_jammed_test_set.pt', 'all_jammed_jammer_outside_region_test_set.pt'],
    'train_set': 'train_dataset.pt',  # train_dataset.pt # triangle_train_set.pt
    'val_set': 'val_dataset.pt',  # val_dataset.pt # triangle_val_set.pt
    'test_set': 'test_dataset.pt',  # test_dataset.pt # triangle_test_set.pt
    'experiments_folder': 'combined_new/',
    'dataset_path': 'data/train_test_data/fspl/combined_fspl.csv',  # combined_fspl_log_distance.csv',  # combined_urban_area.csv
    'all_env': True,
    'inference': False,
    'save_data': True,
    'reproduce': True,
    'plot_network': False,
    'study': 'feat_engineering'  # dataset, coord_system, feat_engineering, knn_edges
}

# CHECK: did you update...
# feats
# in_channels
# out_features
# norm

# 'additional_features': ['node_states', 'dist_to_centroid', 'sin_azimuth', 'cos_azimuth', 'relative_noise', 'proximity_count', 'clustering_coefficient', 'mean_noise', 'median_noise', 'std_noise', 'range_noise'],  # 'elevation_angle', 'sample_connectivity'

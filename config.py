params = {
    'model': 'Sage',
    'learning_rate': 0.0008648799192603533,
    'weight_decay': 0.00005,
    'batch_size': 8,
    'test_batch_size': 8, # 32
    'dropout_rate': 0.5,
    'num_heads': 4,
    'num_layers': 8,
    'hidden_channels': 256,
    'out_channels': 32,
    'out_features': 2,  # 3 jammer pos (x, y, z) # 5 (r, sin(theta), cos(theta), sin(phi), cos(phi))
    'max_epochs': 800,
    '3d': False,
    'coords': 'cartesian',  # opts: 'polar', 'cartesian'
    'required_features': ['node_positions', 'node_noise'],  # node_positions, polar_coordinates, node_noise, node_rssi
    'additional_features': ['mean_noise', 'std_noise', 'range_noise', 'dist_to_centroid', 'sin_azimuth', 'cos_azimuth'],
    'num_neighbors': 'fc',
    'edges': 'knn',  # opts: 'knn', 'proximity'
    'norm': 'minmax',  # opts: 'minmax', 'unit_sphere'
    'activation': False,
    'max_nodes': 300,
    'filtering_proportion': 0.5,
    'grid_meters': 1,
    'ds_method': 'time_window_avg', # time_window_avg, noise
    'experiments_folder': 'dynamic/interpolat/',
    'dataset_path': 'data/modified_interpolated_controlled_path_data_1000.pkl',  # combined_fspl_log_distance.csv',  # combined_urban_area.csv
    'test_sets': ['guided_path_data_test_set.csv', 'linear_path_data_test_set.csv'],
    'dynamic': True,
    'train_per_class': False,
    'all_env_data': False,
    'inference': False,
    'reproduce': True,
    'plot_network': False,
    'study': 'downsampling',  # dataset, coord_system, feat_engineering, knn_edges, downsampling
    'val_discrite_coeff': 0.25,
    'test_discrite_coeff': 0.1,  # disable discritization -> step_size = 1
    'num_workers': 16,
    'aug': ['crop']
}

# 'test_sets': ['test_dataset.pt', 'circle_test_set.pt', 'triangle_test_set.pt', 'rectangle_test_set.pt', 'random_test_set.pt', 'circle_jammer_outside_region_test_set.pt', 'triangle_jammer_outside_region_test_set.pt', 'rectangle_jammer_outside_region_test_set.pt', 'random_jammer_outside_region_test_set.pt', 'all_jammed_test_set.pt', 'all_jammed_jammer_outside_region_test_set.pt'],
#     'train_set': 'all_jammed_jammer_outside_region_train_set.pt',  # train_dataset.pt # all_jammed_train_set.pt # all_jammed_jammer_outside_region_train_set.pt
#     'val_set': 'all_jammed_jammer_outside_region_val_set.pt',  # val_dataset.pt # all_jammed_val_set.pt
#     'test_set': 'all_jammed_jammer_outside_region_test_set.pt',  # test_dataset.pt # all_jammed_test_set.pt
#       'topology': 'all_jammed_jammer_outside_region',

# CHECK: did you update...
# feats
# in_channels
# out_features
# norm

# 'additional_features': ['node_states', 'dist_to_centroid', 'sin_azimuth', 'cos_azimuth', 'relative_noise', 'proximity_count', 'clustering_coefficient', 'mean_noise', 'median_noise', 'std_noise', 'range_noise'],  # 'elevation_angle', 'sample_connectivity'
# 'in_channels': 9,  # x, y, z, rssi # r, sin(theta), cos(theta), sin(phi), cos(phi)), rssi


# 'model': 'Sage',
# 'learning_rate': 0.0008648799192603533,
# 'weight_decay': 0.00005,
# 'batch_size': 8,
# 'dropout_rate': 0.5,
# 'num_heads': 4,
# 'num_layers': 8,
# 'hidden_channels': 256,
# 'out_channels': 32,

# 'model': 'GAT',
# 'learning_rate': 0.0006703373871168542,
# 'weight_decay': 0.00005,
# 'batch_size': 16,
# 'dropout_rate': 0.5,
# 'num_heads': 4,
# 'num_layers': 8,
# 'hidden_channels': 512,
# 'out_channels': 512,
#
# 'model': 'MLP',
# 'learning_rate': 0.0003539362244375827,
# 'weight_decay': 0.00005,
# 'batch_size': 16,
# 'dropout_rate': 0.5,
# 'num_heads': 2,
# 'num_layers': 8,
# 'hidden_channels': 128,
# 'out_channels': 64,
#
# 'model': 'GCN',
# 'learning_rate': 0.0004815321758405571,
# 'weight_decay': 0.00005,
# 'batch_size': 32,
# 'dropout_rate': 0.5,
# 'num_heads': 8,
# 'num_layers': 2,
# 'hidden_channels': 512,
# 'out_channels': 256,

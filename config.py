params = {
    'model': 'GAT',
    'learning_rate': 0.0006703373871168542,
    'weight_decay': 0.00001,
    'test_batch_size': 8,
    'batch_size': 8,
    'dropout_rate': 0.5,
    'num_heads': 4,
    'num_layers': 8,
    'hidden_channels': 128,
    'out_channels': 128,
    'out_features': 5,  # 3 jammer pos (x, y, z) # 5 (r, sin(theta), cos(theta), sin(phi), cos(phi))
    'max_epochs': 300,
    '3d': True,
    'coords': 'polar',  # opts: 'polar', 'cartesian'
    'required_features': ['node_positions', 'node_noise'],  # node_positions, polar_coordinates, node_noise, node_rssi
    'additional_features': ['weighted_centroid_radius','weighted_centroid_sin_theta', 'weighted_centroid_cos_theta', 'weighted_centroid_sin_az', 'weighted_centroid_cos_az', 'dist_to_wcl', 'median_noise', 'max_noise', 'noise_differential', 'vector_x', 'vector_y', 'vector_z', 'rate_of_change_signal_strength'], #'distance_weighted_signal_strength'],
    'num_neighbors': 3,
    'edges': 'knn',  # opts: 'knn', 'proximity'
    'norm': 'unit_sphere',  # opts: 'minmax', 'unit_sphere'
    'activation': False,
    'max_nodes': 1000,
    'filtering_proportion': 0.1,
    'grid_meters': 1,
    'ds_method': 'noise', # time_window_avg, noise
    'experiments_folder': '3d/downsampling/hybrid/500_90/',
    'dataset_path': 'data/controlled_path_data_1000_3d_original.pkl', # 'data/controlled_path_data_1000_3d_original.pkl',  # train_test_data/combined_ua_sua_data.csv',  # combined_urban_area.csv # linear_path_static_new_1to10k_updated_aoa.pkl
    'test_sets': ['guided_path_data_test_set.csv', 'linear_path_data_test_set.csv'],
    'dynamic': True,
    'downsample': True,
    'train_per_class': True,
    'all_env_data': False,
    'inference': False,
    'reproduce': True,
    'plot_network': False,
    'study': 'dataset',  # dataset, coord_system, feat_engineering, knn_edges, downsampling
    'val_discrite_coeff': 0.1, # for testing set to 0.4
    'test_discrite_coeff': 0.1,  # disable discritization -> step_size = 1
    'num_workers': 16,
    'pooling': 'max',
    'aug': ['drop_node'],
    'sn_noise': 'weighted', # "weighted" "max_data"
    'sn_edges': 'directed', # "undirected" "directed"
    'pool_weight_rep': "gr_sn" # "fgr_fgr" "gr_gr" "gr_sn"
}

# 'static/newr/pna/min/'

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
# 'weight_decay': 0.00001,
# 'test_batch_size': 8,
# 'batch_size': 8,
# 'dropout_rate': 0.5,
# 'num_heads': 4,
# 'num_layers': 8,
# 'hidden_channels': 256,
# 'out_channels': 32,
#
# 'model': 'GAT',
# 'learning_rate': 0.0006703373871168542,
# 'weight_decay': 0.00001,
# 'test_batch_size': 8,
# 'batch_size': 8,
# 'dropout_rate': 0.5,
# 'num_heads': 4,
# 'num_layers': 8,
# 'hidden_channels': 128,
# 'out_channels': 128,
#
# 'model': 'MLP',
# 'learning_rate': 0.0003539362244375827,
# 'weight_decay': 0.00001,
# 'test_batch_size': 16,
# 'batch_size': 16,
# 'dropout_rate': 0.5,
# 'num_heads': 2,
# 'num_layers': 8,
# 'hidden_channels': 128,
# 'out_channels': 64,
#
# 'model': 'GCN',
# 'learning_rate': 0.0005815321758405571,
# 'weight_decay': 0.00001,
# 'test_batch_size': 32,
# 'batch_size': 32,
# 'dropout_rate': 0.5,
# 'num_heads': 8,
# 'num_layers': 2,
# 'hidden_channels': 512,
# 'out_channels': 256,


# 'model': 'PNA',
# 'learning_rate': 0.001,
# 'weight_decay': 0.00001,
# 'test_batch_size': 128,
# 'batch_size': 128,
# 'dropout_rate': 0.5,
# 'num_heads': 8,
# 'num_layers': 6,
# 'hidden_channels': 64,
# 'out_channels': 64,

# 'model': 'GPS',
# 'learning_rate': 0.0001,
# 'weight_decay': 0.00001,
# 'test_batch_size': 32,
# 'batch_size': 32,
# 'dropout_rate': 0.0,
# 'num_heads': 4,
# 'num_layers': 10,
# 'hidden_channels': 64,
# 'out_channels': 64,
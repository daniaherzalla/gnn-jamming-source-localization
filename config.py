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
    'feats': 'polar',  # opts: 'polar', 'cartesian'
    'in_channels': 3,  # 4 drone pos (x, y, z) + rssi # 6 (r, sin(theta), cos(theta), sin(phi), cos(phi)) + rssi
    'num_layers': 4,
    'hidden_channels': 256,
    'out_channels': 64,
    'out_features': 2,  # 3 jammer pos (x, y, z) # 5 (r, sin(theta), cos(theta), sin(phi), cos(phi))
    'edges': 'knn',  # opts: 'knn', 'proximity'
    'norm': 'unit_sphere',  # opts: 'zscore', 'minmax', 'unit_sphere'
    'dataset_path': '/home/dania/Downloads/dataset/random/random.csv',  # data/static_swarm_3d.csv # /home/dania/Downloads/dataset/random/random.csv
    'train_path': 'data/train.gzip',
    'val_path': 'data/validation.gzip',
    'test_path': 'data/test.gzip',
    'inference': False,
    'trial_num': 1
}

# params = {
#     'model': 'GATv2',
#     'learning_rate': 0.0034415,
#     'weight_decay': 0.00673,
#     'batch_size': 32,
#     'dropout_rate': 0.11085,
#     'num_heads': 2,
#     'max_epochs': 200,
#     'seed': 42,  # opts: 42, 1, 23
#     '3d': False,
#     'feats': 'cartesian',  # opts: 'polar', 'cartesian'
#     'in_channels': 3,  # 4 drone pos (x, y, z) + rssi # 6 (r, sin(theta), cos(theta), sin(phi), cos(phi)) + rssi
#     'num_layers': 4,
#     'hidden_channels': 256,
#     'out_channels': 64,
#     'out_features': 2,  # 3 jammer pos (x, y, z) # 5 (r, sin(theta), cos(theta), sin(phi), cos(phi))
#     'edges': 'knn',  # opts: 'knn', 'proximity'
#     'norm': 'minmax',  # opts: 'zscore', 'minmax', 'unit_sphere'
#     'dataset_path': '/home/dania/Downloads/dataset/random/random.csv',  # data/static_swarm_3d.csv # /home/dania/Downloads/dataset/random/random.csv
#     'train_path': 'data/train.gzip',
#     'val_path': 'data/validation.gzip',
#     'test_path': 'data/test.gzip',
#     'inference': False,
#     'trial_num': 1
# }

# CHECK: did you update...
# feats
# in_channels
# out_features
# norm

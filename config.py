# params = {
#     'description': '9 features, prox. edges, attention pooling, dropout last layer',
#     'learning_rate': 0.009939984895945914,
#     'weight_decay': 0.017519805495758278,
#     'batch_size': 32,
#     'dropout_rate': 0.20035699277586444,
#     'num_heads': 4,
#     'patience': 30,
#     'max_epochs': 200,
#     'dataset_path': 'data/static_swarm_3d.csv',
#     'train_path': 'data/train.gzip',
#     'val_path': 'data/validation.gzip',
#     'test_path': 'data/test.gzip',
#     'inference': False
# }


params = {
    'model': 'GATv2',
    'learning_rate': 0.005,
    'weight_decay': 0.0001,
    'batch_size': 16,
    'dropout_rate': 0.6,
    'num_heads': 8,
    'max_epochs': 200,
    'seed': 42,  # opts: 42, 1, 23
    'feats': 'polar',  # opts: 'polar', 'cartesian'
    'in_channels': 6,  # 4 drone pos (x, y, z) + rssi # 6 (r, sin(theta), cos(theta), sin(phi), cos(phi)) + rssi
    'num_layers': 4,
    'hidden_channels': 256,
    'out_channels': 64,
    'out_features': 5,  # 3 jammer pos (x, y, z) # 5 (r, sin(theta), cos(theta), sin(phi), cos(phi))
    'edges': 'knn',  # opts: 'knn', 'proximity'
    'norm': 'minmax',  # opts: 'zscore', 'minmax'
    'dataset_path': 'data/static_swarm_3d.csv',
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

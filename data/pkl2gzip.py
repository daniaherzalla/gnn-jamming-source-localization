import pickle
import gzip


def pickle_to_gzip(input_file, output_file):
    # Load data from pickle file
    with open(input_file, 'rb') as f:
        data = pickle.load(f)

    # Write data to gzip file
    with gzip.open(output_file, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


pickle_to_gzip('validation.pkl', 'data/validation.gzip')

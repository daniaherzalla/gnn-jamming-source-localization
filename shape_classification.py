import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import logging
from custom_logging import setup_logging
from config import params

setup_logging()



def save_classifier(classifier, filename):
    joblib.dump(classifier, filename)


def safe_convert_list(row, data_type):
    try:
        if data_type == 'node_positions':
            result = row.strip('[').strip(']').split('], [')
            return [[float(num) for num in elem.split(', ')] for elem in result]
        else:
            raise ValueError("Unsupported data type for conversion")
    except Exception as e:
        logging.error(f"Error converting data: {e}")
        return []  # Return an empty list in case of an error

def load_data(file_path, data_type):
    """Load and convert data from CSV file."""
    data = pd.read_csv(file_path)
    data['node_positions'] = data['node_positions'].apply(lambda x: safe_convert_list(x, data_type))
    return data

def calculate_perimeter(positions):
    positions = np.vstack(positions + [positions[0]])  # Close the polygon
    return np.sum(np.linalg.norm(positions[1:] - positions[:-1], axis=1))

def calculate_polygon_area(coords):
    x, y = zip(*coords)
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def engineer_features(data):
    data['perimeter'] = data['node_positions'].apply(calculate_perimeter)
    data['area'] = data['node_positions'].apply(calculate_polygon_area)
    return data[['perimeter', 'area']]

def train_classifier(X, y):
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X, y)
    return classifier

def evaluate_model(classifier, X_test, y_test):
    y_pred = classifier.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("Test Accuracy: ", accuracy_score(y_test, y_pred))

def main():
    train_data = load_data('/home/dania/gnn-jamming-source-localization/experiments_datasets/datasets/combined_new/train_dataset.csv', 'node_positions')
    test_data = load_data('/home/dania/gnn-jamming-source-localization/experiments_datasets/datasets/combined_new/test_dataset.csv', 'node_positions')

    X_train = engineer_features(train_data)
    X_test = engineer_features(test_data)
    y_train = train_data['node_placement']
    y_test = test_data['node_placement']
    model = train_classifier(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_classifier(model, 'trained_shape_classifier.pkl')  # Save the classifier


if __name__ == "__main__":
    main()

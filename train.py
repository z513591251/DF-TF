from __future__ import print_function
import numpy as np
import argparse
import os
from Filereader import *
from DNAFeature import *
from deepforest import CascadeForestRegressor

# Argument parsing
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Input file for training')
    parser.add_argument('--file', type=str, required=True, help="Path to the input file")
    return parser.parse_args()

def check_file_exists(file_path):
    """Check if the file exists."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

def generate_features(train_seq):
    """Generate features from the training sequences."""
    features = [
        PDMNOnehot(train_seq),
        PDDNOnehot(train_seq),
        DNAComposition(train_seq, 1),
        DNAComposition(train_seq, 2),
        DNAComposition(train_seq, 3),
        DNAComposition(train_seq, 4),
        DNAComposition(train_seq, 5),
        DNAshape(train_seq, MGW),
        DNAshape(train_seq, ProT),
        DNAshape(train_seq, Roll1),
        DNAshape(train_seq, HelT1),
        RFHC(train_seq)
    ]
    return np.hstack(features)

def train_model(x_train, y_train):
    """Train the CascadeForestRegressor model."""
    model = CascadeForestRegressor(random_state=1, use_predictor=1, predictor='lightgbm', backend='sklearn')
    model.fit(x_train, y_train)
    return model

def save_model(model, dirname='model'):
    """Save the trained model."""
    model.save(dirname=dirname)

def main():
    # Parse command line arguments
    args = parse_arguments()
    file_path = args.file

    # Check if file exists
    check_file_exists(file_path)

    # Read training data
    train_seq, train_value = readFile(file_path)

    print('Generating features..........................................')

    # Generate features
    x_train = generate_features(train_seq)

    # Target values
    y_train = np.array(train_value)

    print('Training model..............................................')

    # Train the model
    model = train_model(x_train, y_train)

    # Save the trained model
    save_model(model)

    print('Model training completed and saved.')

if __name__ == "__main__":
    main()
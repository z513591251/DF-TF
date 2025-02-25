import numpy as np
import argparse
from Filereader import readFile, readFasta
from DNAFeature import *
from deepforest import CascadeForestRegressor
from statsmodels.distributions.empirical_distribution import ECDF

# Command-line argument parsing
def parse_arguments():
    parser = argparse.ArgumentParser(description='Input files')
    parser.add_argument('--train', type=str, required=True, help="Path to the training data file")
    parser.add_argument('--test', type=str, required=True, help="Path to the test data file")
    return parser.parse_args()

# Main function to execute the prediction pipeline
def main():
    # Parse command-line arguments
    args = parse_arguments()
    file_train = args.train
    file_test = args.test

    # Reading training and test data
    train_seq, train_value = readFile(file_train)
    test_seq, test_id = readFasta(file_test)

    print('Loading the model............................................')
    # Loading the pre-trained model
    model = CascadeForestRegressor()
    model.load(dirname='model')

    # Create ECDF for the training values
    ecdf = ECDF(train_value)

    result_lines = []  # Store the result rows

    # Iterate over all test sequences
    for sample_idx, sample in enumerate(test_seq):
        print(f"Processing sample {sample_idx + 1} of {len(test_seq)}...")

        _seq = []
        
        # Generate all subsequences of the same length as the training sequence
        subsequence_length = len(train_seq[0])
        for start_idx in range(len(sample) - subsequence_length + 1):
            subseq = sample[start_idx:start_idx + subsequence_length]
            _seq.append(subseq)

        # Feature extraction
        x_test = extract_features(_seq)

        # Predict using the pre-trained model
        print('Predicting..............................................')
        predicted = model.predict(x_test)

        # Get ECDF value for each predicted score using the training set ECDF
        predicted_ecdf_values = ecdf(predicted)

        # Get top 10 predictions based on ECDF (highest ECDF values)
        top_indices = np.argsort(predicted_ecdf_values)[::-1][:10]
        top_scores = predicted_ecdf_values[top_indices]
        top_positions = top_indices
        top_motifs = [_seq[i] for i in top_indices]

        # Prepare result for this sequence (one row per sequence)
        result_row = [f"{position},{score},{motif}" for position, score, motif in zip(top_positions, top_scores, top_motifs)]
        
        # Join the top 10 predictions with commas and add to the result list
        result_lines.append(','.join(result_row))

    # Save results to a CSV file
    save_results(result_lines)

# Feature extraction from subsequences
def extract_features(subsequences):
    x_1 = PDMNOnehot(subsequences)
    x_2 = PDDNOnehot(subsequences)
    x_3 = DNAComposition(subsequences, 1)
    x_4 = DNAComposition(subsequences, 2)
    x_5 = DNAComposition(subsequences, 3)
    x_6 = DNAComposition(subsequences, 4)
    x_7 = DNAComposition(subsequences, 5)
    x_8 = DNAshape(subsequences, MGW)
    x_9 = DNAshape(subsequences, ProT)
    x_10 = DNAshape(subsequences, Roll1)
    x_11 = DNAshape(subsequences, HelT1)
    x_12 = RFHC(subsequences)

    # Stack all features into a single matrix
    x_test = np.hstack((x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12))
    return x_test

# Function to save the results to a CSV file
def save_results(result_lines):
    with open('result.csv', 'w') as f:
        # Write header row
        f.write("Position,ECDF_Score,Motif\n")
        
        # Write the result data
        for line in result_lines:
            f.write(line + '\n')

    print("Results saved to result.csv")

# Run the main function if the script is executed
if __name__ == "__main__":
    main()

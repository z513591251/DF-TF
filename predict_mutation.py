from __future__ import print_function
import numpy as np
import argparse
import csv
from Filereader import readFile, readFasta
from DNAFeature import *
from deepforest import CascadeForestRegressor
from statsmodels.distributions.empirical_distribution import ECDF


# Argument parsing
parser = argparse.ArgumentParser(description='Input files')
parser.add_argument('--train', type=str, default=None, help="Path to the training data file")
parser.add_argument('--ref', type=str, default=None, help="Path to the reference data file")
parser.add_argument('--alt', type=str, default=None, help="Path to the alternate data file")
args = parser.parse_args()

file_train = args.train
file_ref = args.ref
file_alt = args.alt

# Read training and test data
(train_seq, train_value) = readFile(file_train)
(test_ref, ref_id) = readFasta(file_ref)
(test_alt, alt_id) = readFasta(file_alt)

print('Loading the model............................................')

# Load the pre-trained model
model = CascadeForestRegressor()
model.load(dirname='model')

# Create ECDF for the training values
ecdf = ECDF(train_value)

# Initialize lists to store results
score_ref, position_ref, motif_ref, sum_ref = [], [], [], []
score_alt, position_alt, motif_alt, sum_alt = [], [], [], []
position_delta = []

# Iterate through the samples
for sample in range(len(test_ref)):
    seq_ref, seq_alt = [], []
    
    # Process reference sequence
    for string_ref in range(len(test_ref[sample]) - len(train_seq[0]) + 1):
        subseq_ref = test_ref[sample][string_ref:string_ref + len(train_seq[0])]
        seq_ref.append(subseq_ref)
    
    # Extract features for reference sequence
    x_test_ref_1 = PDMNOnehot(seq_ref)
    x_test_ref_2 = PDDNOnehot(seq_ref)
    x_test_ref_3 = DNAComposition(seq_ref, 1)
    x_test_ref_4 = DNAComposition(seq_ref, 2)
    x_test_ref_5 = DNAComposition(seq_ref, 3)
    x_test_ref_6 = DNAComposition(seq_ref, 4)
    x_test_ref_7 = DNAComposition(seq_ref, 5)
    x_test_ref_8 = DNAshape(seq_ref, MGW)
    x_test_ref_9 = DNAshape(seq_ref, ProT)
    x_test_ref_10 = DNAshape(seq_ref, Roll1)
    x_test_ref_11 = DNAshape(seq_ref, HelT1)
    x_test_ref_12 = RFHC(seq_ref)
    
    # Stack features for reference sequence
    x_test_ref = np.hstack((
        x_test_ref_1, x_test_ref_2, x_test_ref_3, x_test_ref_4, x_test_ref_5, x_test_ref_6,
        x_test_ref_7, x_test_ref_8, x_test_ref_9, x_test_ref_10, x_test_ref_11, x_test_ref_12
    ))
    
    # Predict with the reference sequence
    print('Predicting for reference sequence..............................................')
    predicted_ref = model.predict(x_test_ref)
    predicted_ecdf_ref = ecdf(predicted_ref)
    
    # Record reference sequence details
    score_ref.append(np.max(predicted_ecdf_ref))
    position_ref.append(np.argmax(predicted_ecdf_ref))
    sum_ref.append(np.sum(predicted_ecdf_ref))
    motif_ref.append(seq_ref[np.argmax(predicted_ecdf_ref)])

    # Process alternate sequence
    for string_alt in range(len(test_alt[sample]) - len(train_seq[0]) + 1):
        subseq_alt = test_alt[sample][string_alt:string_alt + len(train_seq[0])]
        seq_alt.append(subseq_alt)
    
    # Extract features for alternate sequence
    x_test_alt_1 = PDMNOnehot(seq_alt)
    x_test_alt_2 = PDDNOnehot(seq_alt)
    x_test_alt_3 = DNAComposition(seq_alt, 1)
    x_test_alt_4 = DNAComposition(seq_alt, 2)
    x_test_alt_5 = DNAComposition(seq_alt, 3)
    x_test_alt_6 = DNAComposition(seq_alt, 4)
    x_test_alt_7 = DNAComposition(seq_alt, 5)
    x_test_alt_8 = DNAshape(seq_alt, MGW)
    x_test_alt_9 = DNAshape(seq_alt, ProT)
    x_test_alt_10 = DNAshape(seq_alt, Roll1)
    x_test_alt_11 = DNAshape(seq_alt, HelT1)
    x_test_alt_12 = RFHC(seq_alt)
    
    # Stack features for alternate sequence
    x_test_alt = np.hstack((
        x_test_alt_1, x_test_alt_2, x_test_alt_3, x_test_alt_4, x_test_alt_5, x_test_alt_6,
        x_test_alt_7, x_test_alt_8, x_test_alt_9, x_test_alt_10, x_test_alt_11, x_test_alt_12
    ))
    
    # Predict with the alternate sequence
    print('Predicting for alternate sequence..............................................')
    predicted_alt = model.predict(x_test_alt)
    predicted_ecdf_alt = ecdf(predicted_alt)
    
    # Calculate differences between reference and alternate sequences
    delta = np.absolute(predicted_ecdf_alt - predicted_ecdf_ref)
    
    # Record alternate sequence details and position delta
    position_delta.append(np.argmax(delta))
    score_alt.append(np.max(predicted_ecdf_alt))
    position_alt.append(np.argmax(predicted_ecdf_alt))
    sum_alt.append(np.sum(predicted_ecdf_alt))
    motif_alt.append(seq_alt[np.argmax(predicted_ecdf_alt)])

# Convert results to numpy arrays and reshape for CSV output
motif_ref = np.array(motif_ref).reshape(len(motif_ref), 1)
position_ref = np.array(position_ref).reshape(len(position_ref), 1)
score_ref = np.array(score_ref).reshape(len(score_ref), 1)
sum_ref = np.array(sum_ref).reshape(len(sum_ref), 1)

motif_alt = np.array(motif_alt).reshape(len(motif_alt), 1)
position_alt = np.array(position_alt).reshape(len(position_alt), 1)
score_alt = np.array(score_alt).reshape(len(score_alt), 1)
sum_alt = np.array(sum_alt).reshape(len(sum_alt), 1)

position_delta = np.array(position_delta).reshape(len(position_delta), 1)

# Calculate mutation scores
mutation_score_sum = sum_alt - sum_ref
mutation_score_max = score_alt - score_ref

# Combine results into a single array
result = np.hstack((
    motif_ref, position_ref, score_ref, sum_ref, motif_alt, position_alt, score_alt, sum_alt,
    position_delta, mutation_score_sum, mutation_score_max
))

# Write results to CSV
output_file = 'result.csv'
with open(output_file, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    
    # Add header row
    writer.writerow([
        'motif_ref', 'position_ref', 'score_ref', 'sum_ref', 'motif_alt', 'position_alt', 'score_alt',
        'sum_alt', 'position_delta', 'mutation_score_sum', 'mutation_score_max'
    ])
    
    # Write result data
    writer.writerows(result)

print(f"Results have been saved to {output_file}")

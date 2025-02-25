from __future__ import print_function
import numpy as np
import argparse
import matplotlib.pyplot as plt
from Filereader import readFile, readSeq
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

# Read the training and test sequences
(train_seq, train_value) = readFile(file_train)
test_ref = readSeq(file_ref, len(train_seq[0]))
test_alt = readSeq(file_alt, len(train_seq[0]))

# Feature extraction for the reference sequence
x_test_ref_1 = PDMNOnehot(test_ref)
x_test_ref_2 = PDDNOnehot(test_ref)
x_test_ref_3 = DNAComposition(test_ref, 1)
x_test_ref_4 = DNAComposition(test_ref, 2)
x_test_ref_5 = DNAComposition(test_ref, 3)
x_test_ref_6 = DNAComposition(test_ref, 4)
x_test_ref_7 = DNAComposition(test_ref, 5)
x_test_ref_8 = DNAshape(test_ref, MGW)
x_test_ref_9 = DNAshape(test_ref, ProT)
x_test_ref_10 = DNAshape(test_ref, Roll1)
x_test_ref_11 = DNAshape(test_ref, HelT1)
x_test_ref_12 = RFHC(test_ref)

# Combine features for reference sequence
x_test_ref = np.hstack((
    x_test_ref_1, x_test_ref_2, x_test_ref_3, x_test_ref_4, x_test_ref_5, x_test_ref_6,
    x_test_ref_7, x_test_ref_8, x_test_ref_9, x_test_ref_10, x_test_ref_11, x_test_ref_12
))

# Feature extraction for the alternate sequence
x_test_alt_1 = PDMNOnehot(test_alt)
x_test_alt_2 = PDDNOnehot(test_alt)
x_test_alt_3 = DNAComposition(test_alt, 1)
x_test_alt_4 = DNAComposition(test_alt, 2)
x_test_alt_5 = DNAComposition(test_alt, 3)
x_test_alt_6 = DNAComposition(test_alt, 4)
x_test_alt_7 = DNAComposition(test_alt, 5)
x_test_alt_8 = DNAshape(test_alt, MGW)
x_test_alt_9 = DNAshape(test_alt, ProT)
x_test_alt_10 = DNAshape(test_alt, Roll1)
x_test_alt_11 = DNAshape(test_alt, HelT1)
x_test_alt_12 = RFHC(test_alt)

# Combine features for alternate sequence
x_test_alt = np.hstack((
    x_test_alt_1, x_test_alt_2, x_test_alt_3, x_test_alt_4, x_test_alt_5, x_test_alt_6,
    x_test_alt_7, x_test_alt_8, x_test_alt_9, x_test_alt_10, x_test_alt_11, x_test_alt_12
))

# Load the pre-trained model
print('Loading the model............................................')
model = CascadeForestRegressor()
model.load(dirname='model')

# Make predictions for reference and alternate sequences
print('Predicting..............................................')
predicted_ref = model.predict(x_test_ref)
predicted_alt = model.predict(x_test_alt)

# Compute ECDF for the predictions and training values
ecdf = ECDF(train_value)
predicted_ref_ecdf = ecdf(predicted_ref)
predicted_alt_ecdf = ecdf(predicted_alt)
all_ecdf = ecdf(train_value)

# Plotting the results
plt.figure(figsize=(12, 6))

# First subplot: ECDF of predicted values
plt.subplot(1, 2, 1)
plt.plot(predicted_ref_ecdf, linewidth=2, color='lime', linestyle='solid', marker='*')
plt.plot(predicted_alt_ecdf, linewidth=2, color='red', linestyle='solid', marker='*')
plt.ylim(0, 1)
plt.xlabel('Sequence', fontsize=13)
plt.ylabel('Empirical cumulative distribution score', fontsize=13)
plt.tick_params(labelsize=10)
plt.legend(['ref', 'alt'])

# Second subplot: ECDF with vertical lines for predicted values
plt.subplot(1, 2, 2)
plt.plot(np.array(train_value), all_ecdf, linewidth=4, color='cyan')
plt.ylim(0, 1)
plt.vlines(predicted_ref, 0, 1, color='lime', linewidth=2, linestyle='dashed')
plt.vlines(predicted_alt, 0, 1, color='red', linewidth=2, linestyle='dashed')
plt.xlabel('Relative binding affinity', fontsize=13)
plt.ylabel('Empirical cumulative distribution score', fontsize=13)
plt.tick_params(labelsize=10)

# Save the figure as a PDF
plt.savefig('figure.pdf', bbox_inches='tight')


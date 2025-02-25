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
parser.add_argument('--test', type=str, default=None, help="Path to the test data file")
args = parser.parse_args()

file_train = args.train
file_test = args.test

# Read training data and test sequences
train_seq, train_value = readFile(file_train)
test_seq = readSeq(file_test, len(train_seq[0]))

# Feature extraction for the test sequences
x_test_1 = PDMNOnehot(test_seq)
x_test_2 = PDDNOnehot(test_seq)
x_test_3 = DNAComposition(test_seq, 1)
x_test_4 = DNAComposition(test_seq, 2)
x_test_5 = DNAComposition(test_seq, 3)
x_test_6 = DNAComposition(test_seq, 4)
x_test_7 = DNAComposition(test_seq, 5)
x_test_8 = DNAshape(test_seq, MGW)
x_test_9 = DNAshape(test_seq, ProT)
x_test_10 = DNAshape(test_seq, Roll1)
x_test_11 = DNAshape(test_seq, HelT1)
x_test_12 = RFHC(test_seq)

# Combine all features into a single matrix
x_test = np.hstack((
    x_test_1, x_test_2, x_test_3, x_test_4, x_test_5, x_test_6,
    x_test_7, x_test_8, x_test_9, x_test_10, x_test_11, x_test_12
))

# Load the pre-trained model
print('Loading the model............................................')
model = CascadeForestRegressor()
model.load(dirname='model')

# Predict using the model
print('Predicting..............................................')
predicted = model.predict(x_test)

# Create ECDF for the training values
ecdf = ECDF(train_value)
predicted_ecdf = ecdf(predicted)
all_ecdf = ecdf(train_value)

# Plotting the results
plt.figure(figsize=(12, 6))

# Plot predicted ECDF
plt.subplot(1, 2, 1)
plt.plot(predicted_ecdf, linewidth=2, color='lime', linestyle='solid', marker='*')
plt.ylim(0, 1)
plt.xlabel('Sequence', fontsize=13)
plt.ylabel('Empirical cumulative distribution score', fontsize=13)
plt.tick_params(labelsize=10)

# Plot ECDF for training data
plt.subplot(1, 2, 2)
plt.plot(np.array(train_value), all_ecdf, linewidth=4, color='cyan')
plt.ylim(0, 1)
plt.vlines(predicted, 0, 1, color='lime', linewidth=2, linestyle='dashed')
plt.xlabel('Relative binding affinity', fontsize=13)
plt.ylabel('Empirical cumulative distribution score', fontsize=13)
plt.tick_params(labelsize=10)

# Save the figure to a PDF
plt.savefig('figure.pdf', bbox_inches='tight')

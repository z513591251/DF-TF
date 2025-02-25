import re
import numpy as np

# Helper function to open a file and read its contents
def openFile(input_file): 
    with open(input_file) as f:
        contents = f.readlines()
    return contents

# Function to read a file containing sequences and their associated values
def readFile(input_file):
    contents = openFile(input_file)    
    _sequence = [re.split(r'\s+', line)[0] for line in contents]
    _value = [re.split(r'\s+', line)[1] for line in contents]
    
    # Convert values to float
    _value = list(map(float, _value))
    
    # Get sequence lengths
    _length = [len(eachseq) for eachseq in _sequence]   
    
    # Check for consistency in sequence lengths
    if len(set(_length)) == 1:
        return _sequence, _value 
    else:
        print('Warning!!! Inconsistent sequence length')
        return None, None

# Function to read feature file and extract values and features
def readFeature(input_file):
    contents = openFile(input_file)    
    _value = [re.split(r'\s+', line)[0] for line in contents]
    
    # Extract features starting from the third column onwards
    _feature = [re.split(r'\s+', line)[2:] for line in contents]
    
    # Remove any empty elements in the feature list
    _feature = [[i for i in line if len(str(i)) != 0] for line in _feature]
    
    # Convert values to float
    _value = list(map(float, _value))
    
    # Convert features into a numpy array
    _feature = np.array(_feature)
    
    return _value, _feature

# Function to read a sequence file and generate subsequences of a given window size
def readSeq(input_file, windows):
    _seq = []
    contents = openFile(input_file)
    
    # Strip new line character from the first line of the file
    contents[0] = contents[0].strip('\n')
    
    # Extract subsequences with the given window size
    for num in range(len(contents[0]) - windows + 1):
        subseq = contents[0][num:num + windows]
        _seq.append(subseq)
    
    return _seq

# Function to read a FASTA file and extract sequences with their IDs
def readFasta(inpFile):
    _hash = {}
    _seq = []
    _id = []

    # Read lines from the input file
    for line in open(inpFile):
        if line.startswith('>'):
            name = line.replace('>', '').split()[0]
            _id.append(name)
            _hash[name] = ''
        else:
            # Append sequence data to the corresponding name
            _hash[name] += line.replace('\n', '')

    # Collect sequences from the dictionary
    for i in _hash.keys():
        Subseq = _hash[i]
        _seq.append(Subseq)

    return _seq, _id

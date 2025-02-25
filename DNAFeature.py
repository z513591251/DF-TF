"""
Feature Extraction Functions for DNA/RNA Sequences

Functions:
    - PDMNOnehot    : Position-Dependent MonoNucleotides Onehot (DNA or RNA)
    - PDMNDict      : Position-Dependent MonoNucleotides Dictionary (DNA or RNA)
    - PDDNOnehot    : Position-Dependent DiNucleotides Onehot (DNA or RNA)
    - PDDNDict      : Position-Dependent DiNucleotides Dictionary (DNA or RNA)
    - PDTNOnehot    : Position-Dependent TriNucleotides Onehot (DNA or RNA)
    - PDTNDict      : Position-Dependent TriNucleotides Dictionary (DNA or RNA)
    - DNAComposition: DNA Composition
    - GCcount       : The number of Gs and Cs
    - SpeKmercount  : The number of specific k-mer sub-sequence
    - DNPCP         : DiNucleotides PhysicoChemical Properties
    - TNPCP         : TriNucleotides PhysicoChemical Properties
    - DNAshape      : DNA structural features
    - RFHC          : Rings, Functional groups, and Hydrogen bonds Composition
"""
import numpy as np
import itertools
import sys
sys.path.append('.//indices')
from Wordvec import *
from Dinucleotide_indices import *
from Trinucleotide_indices import *
from Shape import *


def PDMNOnehot(inpSeq):
    """Generate one-hot encoding for position-dependent mono-nucleotides."""
    onehot = []
    for seq in inpSeq:
        onehot.append([OnehotMonoNuc.get(base) for base in seq])
    return np.array(onehot).reshape(len(inpSeq), len(seq) * 4)


def PDMNDict(inpSeq):
    """Generate dictionary encoding for position-dependent mono-nucleotides."""
    dict_encoding = []
    for seq in inpSeq:
        dict_encoding.append([DictMonoNuc.get(base) for base in seq])
    return np.array(dict_encoding).reshape(len(inpSeq), len(seq))


def PDDNOnehot(inpSeq):
    """Generate one-hot encoding for position-dependent di-nucleotides."""
    onehot = []
    for seq in inpSeq:
        onehot.append([OnehotDiNuc.get(seq[num:num + 2]) for num in range(len(seq) - 1)])
    return np.array(onehot).reshape(len(inpSeq), (len(seq) - 1) * 16)


def PDDNDict(inpSeq):
    """Generate dictionary encoding for position-dependent di-nucleotides."""
    dict_encoding = []
    for seq in inpSeq:
        dict_encoding.append([DictDiNuc.get(seq[num:num + 2]) for num in range(len(seq) - 1)])
    return np.array(dict_encoding).reshape(len(inpSeq), (len(seq) - 1))


def PDTNOnehot(inpSeq):
    """Generate one-hot encoding for position-dependent tri-nucleotides."""
    onehot = []
    for seq in inpSeq:
        onehot.append([OnehotTriNuc.get(seq[num:num + 3]) for num in range(len(seq) - 2)])
    return np.array(onehot).reshape(len(inpSeq), (len(seq) - 2) * 64)


def PDTNDict(inpSeq):
    """Generate dictionary encoding for position-dependent tri-nucleotides."""
    dict_encoding = []
    for seq in inpSeq:
        dict_encoding.append([DictTriNuc.get(seq[num:num + 3]) for num in range(len(seq) - 2)])
    return np.array(dict_encoding).reshape(len(inpSeq), (len(seq) - 2))


def DNAComposition(inpSeq, kmer):
    """
    Calculate DNA composition for a given k-mer size.
    
    Args:
        inpSeq (list): List of DNA sequences.
        kmer (int): Size of the k-mer (1 for mono, 2 for di, 3 for tri, etc.).
    
    Returns:
        np.ndarray: Composition matrix of shape (n_sequences, 4^kmer).
    """
    composition = []
    kmer_list = list(map(''.join, itertools.product('ATCG', repeat=kmer)))
    for seq in inpSeq:
        segments = [seq[num:num + kmer] for num in range(len(seq) - kmer + 1)]
        composition.append([(segments.count(seg) / (len(seq) - kmer + 1)) for seg in kmer_list])
    return np.array(composition).reshape(len(inpSeq), 4**kmer)


def GCcount(inpSeq, start, end):
    """Count the number of G and C bases in a specified region of each sequence."""
    counts = []
    for seq in inpSeq:
        sub_seq = seq[start - 1:end]
        counts.append(sub_seq.count('G') + sub_seq.count('C'))
    return np.array(counts).reshape(len(inpSeq), 1)


def SpeKmercount(inpSeq, Kmer, start, end):
    """Count the occurrences of a specific k-mer in a specified region of each sequence."""
    counts = []
    length = end - start
    if length < len(Kmer):
        raise ValueError("The length of the sequence should be larger than that of the k-mer subsequence.")
    for seq in inpSeq:
        sub_seq = seq[start - 1:end]
        segments = [sub_seq[num:num + len(Kmer)] for num in range(len(sub_seq) - len(Kmer) + 1)]
        counts.append(segments.count(Kmer))
    return np.array(counts).reshape(len(inpSeq), 1)


def DNPCP(inpSeq, indice, windows):
    """Calculate DiNucleotide PhysicoChemical Properties (DNPCP) with a sliding window."""
    if len(indice.keys()) != 16:
        raise ValueError("Wrong name of physicochemical indices! Please see the Dinucleotide_indices.")
    properties = []
    averages = []
    for seq in inpSeq:
        properties.append([indice.get(seq[num:num + 2]) for num in range(len(seq) - 1)])
    properties = np.array(properties).reshape(len(inpSeq), (len(seq) - 1))
    for prop in properties:
        averages.append([np.mean(prop[num:num + windows]) for num in range(len(prop) - windows + 1)])
    return np.array(averages).reshape(len(properties), len(prop) - windows + 1)


def TNPCP(inpSeq, indice, windows):
    """Calculate TriNucleotide PhysicoChemical Properties (TNPCP) with a sliding window."""
    if len(indice.keys()) != 64:
        raise ValueError("Wrong name of physicochemical indices! Please see the Trinucleotide_indices.")
    properties = []
    averages = []
    for seq in inpSeq:
        properties.append([indice.get(seq[num:num + 3]) for num in range(len(seq) - 2)])
    properties = np.array(properties).reshape(len(inpSeq), (len(seq) - 2))
    for prop in properties:
        averages.append([np.mean(prop[num:num + windows]) for num in range(len(prop) - windows + 1)])
    return np.array(averages).reshape(len(properties), len(prop) - windows + 1)


def DNAshape(inpSeq, indice):
    """Calculate DNA structural features."""
    if len(indice.keys()) != 1024:
        raise ValueError("Wrong name of DNA shape! Please see the Shape.")
    shape = []
    for seq in inpSeq:
        shape.append([indice.get(seq[num:num + 5]) for num in range(len(seq) - 4)])
    return np.array(shape).reshape(len(inpSeq), (len(seq) - 4))


def RFHC(inpSeq):
    """Calculate Rings, Functional groups, and Hydrogen bonds Composition."""
    composition = []
    RFC = {'A': [1, 1, 1],
           'C': [0, 0, 1],
           'G': [1, 0, 0],
           'T': [0, 1, 0]}
    for seq in inpSeq:
        composition.append([RFC.get(base) for base in seq])
    return np.array(composition).reshape(len(inpSeq), len(seq) * 3)
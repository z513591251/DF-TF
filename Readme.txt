DeepForest-Based Modeling of Transcription Factor Binding Specificities and Affinity Changes
======================
This repository provides tools to model transcription factor (TF) binding specificities and predict the impact of sequence variants on binding affinity using the DeepForest framework. The workflow includes training models on HT-SELEX data, predicting TF binding sites, and analyzing sequence mutations.
======================
Installation
Install the required package:
pip install deep-forest
======================
Dataset
The data_humo folder contains HT-SELEX data for 215 transcription factors. For demonstration, we use the CEBPB transcription factor. Example files are provided in the example folder:

Training data: bZIP_CEBPB_TGGACA20NGA_TTRCGC_12_4.txt

Test sequences: remap2022_CEBPB_nr_macs2_hg38_v1_0.fasta

Mutation analysis files: ref_predict.txt, alt_predict.txt
======================
Usage
1. Train a DeepForest Model
Train a model on CEBPB HT-SELEX data:
python train.py --file example/bZIP_CEBPB_TGGACA20NGA_TTRCGC_12_4.txt

2. Predict Binding Sites
Predict potential CEBPB binding sites in genomic sequences using the trained model and empirical cumulative distribution function (ECDF):
python predict_ECDF.py \
  --train example/bZIP_CEBPB_TGGACA20NGA_TTRCGC_12_4.txt \
  --test example/remap2022_CEBPB_nr_macs2_hg38_v1_0.fasta

Output: predict_ECDF.csv lists the top 10 predicted positions, scores, and subsequences for each input sequence.

3. Plot Binding Affinity ECDF
Visualize ECDF values to define binding thresholds:
python plot_ECDF.py \
  --train example/bZIP_CEBPB_TGGACA20NGA_TTRCGC_12_4.txt \
  --test example/CEBPB_plot.txt

Output: plot_ECDF.pdf shows the ECDF curve for binding affinity scores.

4. Predict Mutation Effects
Evaluate how sequence mutations impact CEBPB binding affinity:
python predict_mutation.py \
  --train example/bZIP_CEBPB_TGGACA20NGA_TTRCGC_12_4.txt \
  --ref example/ref_predict.txt \
  --alt example/alt_predict.txt

Output: predict_mutation.csv reports:

Reference/alternate subsequences with highest scores

Total ECDF change and maximum ECDF change caused by mutations

5. Visualize Mutation Impacts
Plot mutation-induced ECDF changes:
python plot_mutation.py \
  --train example/bZIP_CEBPB_TGGACA20NGA_TTRCGC_12_4.txt \
  --ref example/ref_plot.txt \
  --alt example/alt_plot.txt

Output: plot_mutation.pdf illustrates the effect of mutations on binding affinity.
============================
Output Examples
Example results are provided in the results folder:

Binding site predictions: predict_ECDF.csv

ECDF visualization: plot_ECDF.pdf

Mutation analysis: predict_mutation.csv, plot_mutation.pdf




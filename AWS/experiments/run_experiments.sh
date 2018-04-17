#!/bin/bash

####
# Run Experiments from the paper
# Pass in the number of runs and epochs as command line arguments
####

# Numer of runs to repeat for each model configuration
RUNS=$1
# Number of epochs/run
EPOCHS=$2

# Train logistic regression classifiers
echo "TRAINING LOGISTIC REGRESSION..."
python3 logreg_experiments.py --epochs $EPOCHS --runs $RUNS
echo "done."
echo

# Train FFNN experiments
echo "TRAINING FFNN MODELS..."
python3 ffnn_experiments.py --epochs $EPOCHS --runs $RUNS
echo "done."
echo

# Train CINFARNET experiments
echo "TRAINING CINFAET MODELS..."
python3 cifarnet_experiments.py --epochs $EPOCHS --runs $RUNS
echo "done."
echo

echo "KTHXBYE."
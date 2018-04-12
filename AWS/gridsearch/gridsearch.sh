#!/bin/bash

echo "Tuning Logistic Regression models..."
python3 tune_logreg.py
echo "Done."
echo
echo "Tuning FFNN models..."
python3 tune_ffnn.py
echo "Done."
echo
echo "Tuning CIFARNET models..."
python3 tune_cifarnet.py
echo "Done."
echo
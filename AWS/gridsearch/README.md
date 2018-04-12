## Gridsearch to tune hyperparameters _alpha_ and _beta2_

For each of the three models (Logistic Regression, FFNN on MNIST, CNN on CIFAR), we need to tune two learning parameters using a gridsearch. All other hyperparameters and architecture details are specified by the authors. 

To speed up this process, I want to run scripts for hyperparameter tuning on AWS GPUs. This directory is one that I can zip up, `scp` to Amazon's servers, and run there as is. As such, many files in this directory are coppied from the other directories in this repository. 

The idea is that I can simply
- `scp` this directory
- run the `gridsearch.sh` bash script
- `scp` the pickled classifiers and log files back to my computer. 

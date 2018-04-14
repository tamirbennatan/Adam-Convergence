## Experiment scripts

Here are some simple scripts that allow me to run the experiments described in the paper multiple times, and save the results to a `csv` file. This is mosly for convinience, so that I can `scp` this entire directory to AWS and train on their GPU's. 

The python scripts in take the following command line arguments:
- `runs` - how many times to run the models with each of the two optimization methods (AMSGrad, Adam)
- `epochs` - how many epochs per experiment. 
- `batch` (default 128) the batch size. 

For example, I could run:
```bash
python3 ffnn_experiments.py --runs 10 --epochs 20
```
And the script will train the feedforward neural network 10 times with each optimizer, each time for 20 epochs. 

The results are saved in the `experiment_log` directory as csv files. 

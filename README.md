## Reproducing **On the Convergence of Adam and Beyond**

In this project we reproduce the results from the the paper [On the Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ) (Sashank, Kale, Kumar; 2018). 

This project useis the `MNIST` and `CIFAR-10` datasets. The scripts in the `data/` directory will save the `MNIST` and `CIFAR-10` datasets as binary numpy files (`.npy` extention). You will have to [download the CIFAR-10 dataset from here](https://www.cs.toronto.edu/~kriz/cifar.html), and modify the first line of the script `data/CIFAR/load_cifar.py` to process the data.

In the `AWS/` directory you will find a series of scripts which I used to train each of the models on Amazon's EC2 service. These scripts are modifications of the work found in the `Logistic-Regression` and `Neural-Networks` directories.

Finally, you can find the results of our training runs in `results`, and a paper discussing our results in `Report`. 

ಠ‿↼

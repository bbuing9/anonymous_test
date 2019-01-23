# anonymous_test

## Preliminaries
It is tested under Ubuntu Linux 16.04.1 and Python 3.5 environment, and requries Pytorch package to be installed:

* [Pytorch](http://pytorch.org/): Only GPU version (0.4.1) is available.

Reference code: https://github.com/facebookresearch/mixup-cifar10

### 1. Training a network with specified imbalance on CIFAR-100:
```
# dataset: CIFAR-100, minimal class size: 25, imbalance level: 0.5, iteration for searching a decision boundary: 5, ratio between cross entropy loss and mathing loss
python train_boundary.py --model ResNet18_100 --name 'test' -o --num_search 5 --beta 0.1 --num_min 25 --gamma 0.5
```

## Performance evaluation

Loss and average accuracy of train, validation and test are logged as csv file. Please check it in ./results 
Also, the class-wise accuracy is saved in ./results at the epoch with the best validation performance
By using this, one can ge the Q_25, Q_50 and Q_100 as we reported in the paper

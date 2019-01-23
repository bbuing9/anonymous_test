# anonymous_test

## Preliminaries
It is tested under Ubuntu Linux 16.04.1 and Python 3.5 environment, and requries Pytorch package to be installed:
* Reference code: https://github.com/facebookresearch/mixup-cifar10
* [Pytorch](http://pytorch.org/): Only GPU version (0.4.1) is available.

## Arguments

* --lr : control learning rate ( default: 0.1 )
* --resume : resume from checkpoint
* --model : define a model type ( default: ResNet18 )
* --batch-size : batch size ( default: 128 )
* --epoch : total epochs to run ( default: 200 )
* --decay : weight decay ( default=1e-4 )
* --alpha : mixup interpolation coefficient (default: 1)

* --gamma : imbalanced level ( default: 2.0 )
* --num_min : size of minimal class ( default: 25 )
* --num_imb : number of defined minority classes ( default: 50 )
* --warm : starting epoch for applying Boundary-Mixup ( default: 180 )
* --num_search : number of iteration to find a decision boundary ( default: 5 )
* --beta : ratio between cross entropy loss and matching loss ( default: 0.1 )

* --over : sampling from balanced data with replication
* --smote : sampling from balanced data with interpolation within the same class

## Training networks with class imbalanced data

### 1. Training
```
# ratio between loss: 0.1, imbalance level: 2.0, minimal class size: 25, iteration to find a decision boundary: 5, 
python train_boundary.py -o --num_search 5 --beta 0.1 --num_min 25 --gamma 2.0
```
### 2. Evaluation

Loss, accuracy for train, val and test sets are logged as csv files
Also, the class-wise test accuracy is saved at the epoch with best val acc 


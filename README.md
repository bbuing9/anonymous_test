# Mixup-CIFAR10
By [Hongyi Zhang](http://web.mit.edu/~hongyiz/www/), [Moustapha Cisse](https://mine.kaust.edu.sa/Pages/cisse.aspx), [Yann Dauphin](http://dauphin.io/), [David Lopez-Paz](https://lopezpaz.org/).

Facebook AI Research

## Introduction

Mixup is a generic and straightforward data augmentation principle.
In essence, mixup trains a neural network on convex combinations of pairs of
examples and their labels. By doing so, mixup regularizes the neural network to
favor simple linear behavior in-between training examples.

This repository contains the implementation used for the results in
our paper (https://arxiv.org/abs/1710.09412).

## Requirements and Installation
* A computer running macOS or Linux
* For training new models, you'll also need a NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* Python version 3.6
* A [PyTorch installation](http://pytorch.org/)

## Training
Use `python train.py` to train a new model.
Here is an example setting:
```
$ CUDA_VISIBLE_DEVICES=0 python train.py --lr=0.1 --seed=20170922 --decay=1e-4
```

## Arguments

* --resume : turn on(-r)/off for resuming from save checkpoint.
* --over : turn on(-o)/off an oversampling weight to Cross entropy loss
* --type1 : turn on(-t)/off a linear imbalance. If it is turned off, then step imbalance is applied. 
          ( Linear imbalance should be applied with args.ratio = 0.1, args.num_imb = 5 )
* --start_idx : start index of imbalance. E.g., if it is set to 4, then [3500, 4000, 4500, 5000, 500, 1000, 1500, 2000, 2500, 3000] is a list for number of samples.
* --ratio : imbalance ratio between majority and minority. 
* --num_imb : a number of minority. E.g., if it is set to 5, then a half of class become minority
* --alpha : a hyperparameter of a beta distribution for mixup. Default is 1. 
```
$ CUDA_VISIBLE_DEVICES=0 python3 train_unbal.py --name 'linear_start_4_overmixup_imb5_alpha2.0_trial1' -t -o --start_idx 4 --alpha 2.0 --num_imb 5 --ratio 0.1 
```

## License

This project is CC-BY-NC-licensed.

## Acknowledgement
The CIFAR-10 reimplementation of _mixup_ is adapted from the [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar) repository by [kuangliu](https://github.com/kuangliu).

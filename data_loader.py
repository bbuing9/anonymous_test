import torch
import os
import numpy.random as nr
import numpy as np
import bisect

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler

num_test_samples_cifar100 = [100] * 100

def get_val_test_data(dataset, num_sample_per_class, shuffle = False, random_seed = 0):
    """
    Return a list of indices for validation and test from a dataset.
    Input: A test dataset (e.g., CIFAR-10)
    Output: validation_list and test_list
    """
    length = dataset.__len__()
    num_samples = num_sample_per_class[0] # Suppose that all classes have the same number of test samples
    val_list = []
    test_list = []
    indices = list(range(0,length))
    if shuffle:
        nr.shuffle(indices)
    for i in range(0, length):
        index = indices[i]
        _, label = dataset.__getitem__(index)
        if num_sample_per_class[label] > (9 * num_samples / 10):
            val_list.append(index)
            num_sample_per_class[label] -= 1
        else:
            test_list.append(index)
            num_sample_per_class[label] -= 1

    return val_list, test_list


def get_imbalanced_data(dataset, num_sample_per_class, shuffle = False, random_seed = 0):
    """
    Return a list of imbalanced indices from a dataset.
    Input: A dataset (e.g., CIFAR-10), num_sample_per_class: list of integers
    Output: imbalanced_list
    """
    length = dataset.__len__()
    selected_list = []
    indices = list(range(0,length))
    if shuffle:
        nr.shuffle(indices)
    for i in range(0, length):
        index = indices[i]
        _, label = dataset.__getitem__(index)
        if num_sample_per_class[label] > 0:
            selected_list.append(index)
            num_sample_per_class[label] -= 1

    return selected_list


def get_oversampling_cifar100(num_sample_per_class, smote, batch_size, TF_train, TF_test,
                                data_root='/tmp/public_dataset/pytorch', **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar100-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building CIFAR-100 CV data loader with {} workers".format(num_workers))
    ds = []
    train_cifar = datasets.CIFAR100(root=data_root, train=True, download=False, transform=TF_train)

    targets = np.array(train_cifar.train_labels)
    classes, class_counts = np.unique(targets, return_counts=True)
    nb_classes = len(classes)

    imbal_class_counts = [int(i) for i in num_sample_per_class]
    class_indices = [np.where(targets == i)[0] for i in range(nb_classes)]

    imbal_class_indices = [class_idx[:class_count] for class_idx, class_count in zip(class_indices, imbal_class_counts)]
    imbal_class_indices = np.hstack(imbal_class_indices)

    train_cifar.train_labels = targets[imbal_class_indices]
    train_cifar.train_data = train_cifar.train_data[imbal_class_indices]

    assert len(train_cifar.train_labels) == len(train_cifar.train_data)

    targets = np.array(train_cifar.train_labels)
    imb_cifar_data = train_cifar.train_data
    class_max = max(num_sample_per_class)
    aug_data = []
    aug_label = []
    for k in range(nb_classes):
        indices = np.where(targets == k)[0]
        class_data = imb_cifar_data[indices]
        class_len = len(indices)
        class_dist = np.zeros((class_len, class_len))
        # Augmentation with SMOTE ( k-nearest )
        if smote:
            for i in range(class_len):
                for j in range(class_len):
                    class_dist[i, j] = np.linalg.norm(class_data[i] - class_data[j])
            sorted_idx = np.argsort(class_dist)

            for i in range(class_max - class_len):
                lam = nr.uniform(0, 1)
                row_idx = i % class_len
                col_idx = int((i - row_idx) / class_len)
                new_data = np.round(lam * class_data[row_idx] + (1 - lam) * class_data[sorted_idx[row_idx, 1 + col_idx]])
                aug_data.append(new_data.astype('uint8'))
                aug_label.append(k)
        # Augmentation with naive oversampling
        else:
            for i in range(class_max - class_len):
                rand_idx = np.random.randint(0, class_len, size=1)[0]
                new_data = class_data[rand_idx]
                aug_data.append(new_data.astype('uint8'))
                aug_label.append(k)

    aug_data = np.array(aug_data)
    aug_label = np.array(aug_label)
    train_cifar.train_labels = np.concatenate((targets, aug_label), axis=0)
    train_cifar.train_data = np.concatenate((imb_cifar_data, aug_data), axis=0)
    print("Augmented data num = {}".format(len(aug_label)))
    print(train_cifar.train_data.shape)
    train_in_loader = torch.utils.data.DataLoader(train_cifar, batch_size=batch_size, shuffle=True, **kwargs)
    ds.append(train_in_loader)

    test_cifar = datasets.CIFAR100(root=data_root, train=False, download=False, transform=TF_test)
    val_idx, test_idx = get_val_test_data(test_cifar, num_test_samples_cifar100)
    val_loader = torch.utils.data.DataLoader(test_cifar, batch_size=100,
                                             sampler=SubsetRandomSampler(val_idx), **kwargs)
    test_loader = torch.utils.data.DataLoader(test_cifar, batch_size=100,
                                              sampler=SubsetRandomSampler(test_idx), **kwargs)
    ds.append(val_loader)
    ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds

    return ds


def get_imbalanced_cifar100(num_sample_per_class, batch_size, TF_train, TF_test, data_root='/tmp/public_dataset/pytorch',
                           train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar100-data'))
    num_workers = kwargs.setdefault('num_workers', 8)
    kwargs.pop('input_size', None)
    print("Building CIFAR-100 CV data loader with {} workers".format(num_workers))
    ds = []
    train_cifar = datasets.CIFAR100(root=data_root, train=True, download=True, transform=TF_train)
    train_in_idx = get_imbalanced_data(train_cifar, num_sample_per_class)
    train_in_loader = torch.utils.data.DataLoader(train_cifar, batch_size=batch_size,
                                                  sampler=SubsetRandomSampler(train_in_idx), **kwargs)
    ds.append(train_in_loader)

    test_cifar = datasets.CIFAR100(root=data_root, train=False, download=False, transform=TF_test)
    val_idx, test_idx = get_val_test_data(test_cifar, num_test_samples_cifar100)
    val_loader = torch.utils.data.DataLoader(test_cifar, batch_size=100,
                                             sampler=SubsetRandomSampler(val_idx), **kwargs)
    test_loader = torch.utils.data.DataLoader(test_cifar, batch_size=100,
                                              sampler=SubsetRandomSampler(test_idx), **kwargs)
    ds.append(val_loader)
    ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds

    return ds
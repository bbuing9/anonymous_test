#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.
from __future__ import print_function

import argparse
import csv
import os

import numpy as np
import torch
from torch.autograd import Variable, grad
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import data_loader
import models
from utils import progress_bar, make_imb_data

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--model', default="ResNet18", type=str,
                    help='model type (default: ResNet18)')
parser.add_argument('--name', default='0', type=str, help='name of run')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--batch-size', default=128, type=int, help='batch size')
parser.add_argument('--epoch', default=200, type=int,
                    help='total epochs to run')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='use standard augmentation (default: True)')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--alpha', default=1., type=float,
                    help='mixup interpolation coefficient (default: 1)')
parser.add_argument('--gamma', default=2.0, type=float, help='imbalanced level')
parser.add_argument('--num_imb', default=34, type=int, help='number of minority classes')
parser.add_argument('--num_min', default=25, type=int, help='number of samples of minimal class')
parser.add_argument('--warm', default=180, type=int, help='starting epoch for applying Boundary Mixup')
parser.add_argument('--num_search', default=5, type=int, help='number of iteration to find a decision boundary')
parser.add_argument('--beta', default=0.01, type=float, help='a hyper-parameter for matching loss')

parser.add_argument('--over', '-o', action='store_true', help='oversampling')
parser.add_argument('--smote', '-s', action='store_true', help='oversampling')
parser.add_argument('--binary', '-binary', action='store_true', help='find a lambda with binary search')

args = parser.parse_args()
use_cuda = torch.cuda.is_available()

best_acc = 0  # best test accuracy
best_auc = 0  # best test auc
best_val =0  # best validation accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if args.seed != 0:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Data
print('==> Preparing data..')
if args.augment:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409),
                             (0.2009, 0.1984, 0.2023)),
    ])
else:
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409),
                             (0.2009, 0.1984, 0.2023)),
    ])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2009, 0.1984, 0.2023)),
])

# Imbalance
num_class = 100
num_samples = 500.0
num_samples_per_class = make_imb_data(num_samples, args.num_min, num_class, args.gamma)

if args.over:
    trainloader, valiloader, testloader = data_loader.get_oversampling_cifar100(num_samples_per_class, False,
                                                                              args.batch_size,
                                                                              transform_train,
                                                                              transform_test,
                                                                              data_root='./data')
else:
    trainloader, valiloader, testloader = data_loader.get_imbalanced_cifar100(num_samples_per_class,
                                                                                 args.batch_size,
                                                                                 transform_train,
                                                                                 transform_test,
                                                                                 data_root='./data')


# Model
print('==> Building model..')
net = models.__dict__[args.model]()

if use_cuda:
    net.cuda()
    print(torch.cuda.device_count())
    cudnn.benchmark = True
    print('Using CUDA..')

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)

# Resume
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7' + args.name )
    net.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
    rng_state = checkpoint['rng_state']
    torch.set_rng_state(rng_state)

if not os.path.isdir('results'):
    os.mkdir('results')
logname = ('results/log_' + net.__class__.__name__ + '_' + args.name + '.csv')

criterion = nn.CrossEntropyLoss(reduction='none')
mse_loss = nn.MSELoss()


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns pairs of inputs and targets, and a mixing ratio lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    x_mix = x[index, :]
    y_a, y_b = y, y[index]
    return x, x_mix, y_a, y_b, lam


def correct_random(inputs, targets):
    net.eval()
    batch_size = inputs.size(0)
    selected_idx = torch.LongTensor(batch_size).cuda()
    arange = torch.arange(batch_size).cuda()

    outputs, _ = net(inputs)
    prob = F.softmax(outputs, dim=1)
    _, max_idx = torch.max(prob, 1)
    pred = max_idx.eq(targets.data)
    wrong = max_idx.ne(targets.data)

    pred_idx = pred.nonzero()
    pred_idx = pred_idx.reshape(pred_idx.size(0))
    num_correct = pred_idx.size(0)

    # Do not apply mixup for misclassified samples
    selected_idx[wrong] = arange[wrong]

    # Generate index for cross-class
    correct_targets = targets[pred_idx]
    mix_idx = torch.LongTensor(num_correct).cuda()

    for i in range(num_correct):
        sample_class = correct_targets[i]
        possible_samples = correct_targets.ne(sample_class).nonzero()
        possible_samples = possible_samples.reshape(possible_samples.size(0))

        if possible_samples.size(0) == 0:
            mix_idx[i] = i
        else:
            mix_idx[i] = possible_samples[torch.randint(0, possible_samples.size(0), [1]).long()]

    selected_idx[pred_idx] = pred_idx[mix_idx]

    net.train()

    return selected_idx, prob, pred


def linear_solv(x_a, x_b, y_a, y_b, inputs_a, inputs_b):
    # Find a solution x* s.t. g(x)=ax+b=0 (in here, g(x)=f_i(x)-f_j(x))
    eps = 1e-10
    y_a, y_b = y_a.reshape(y_a.size(0), 1, 1, 1), y_b.reshape(y_a.size(0), 1, 1, 1)
    x_sol = -1 * (x_a*y_b - x_b*y_a) / (y_a - y_b + eps)

    return x_sol


def lam_binary_search(inputs, targets, num_iter, selective_idx, prob, pred):
    net.eval()
    eps = 1e-10
    batch_size = inputs.size(0)
    lams = torch.ones(batch_size, 1, 1, 1).cuda()
    dist = 0

    # For correctly classified samples, find decision boundary with a false position method
    pred_idx = pred.nonzero()
    pred_idx = pred_idx.reshape(pred_idx.size(0))
    num_correct = pred_idx.size(0)
    mix_idx = selective_idx[pred_idx]

    if mix_idx.size(0) > 0:
        # Two anchors for binary search
        inputs_a, targets_a = inputs[pred_idx], targets[pred_idx]
        inputs_b, targets_b = inputs[mix_idx], targets[mix_idx]
        prob_a, prob_b = prob[pred_idx], prob[mix_idx]

        # Find f_i and f_j
        true_onehot = torch.ByteTensor(num_correct, num_class).cuda()
        shuffle_onehot = torch.ByteTensor(num_correct, num_class).cuda()
        true_onehot.zero_()
        shuffle_onehot.zero_()

        true_onehot.scatter_(1, targets_a.unsqueeze(1).data, 1)
        shuffle_onehot.scatter_(1, targets_b.unsqueeze(1).data, 1)

        # f_i - f_j of two points
        x_a, x_b = inputs_a, inputs_b
        y_a = torch.masked_select(prob_a, true_onehot) - torch.masked_select(prob_a, shuffle_onehot)
        y_b = torch.masked_select(prob_b, true_onehot) - torch.masked_select(prob_b, shuffle_onehot)

        inputs_lam = linear_solv(x_a, x_b, y_a, y_b, inputs_a, inputs_b)
        outputs_lam, _ = net(inputs_lam)
        prob_lam = F.softmax(outputs_lam, dim=1).data
        y_lam = torch.masked_select(prob_lam, true_onehot) - torch.masked_select(prob_lam, shuffle_onehot)

        for i in range(num_iter):
            left_idx = y_lam.ge(0)
            right_idx = 1 - left_idx
            x_a2, x_b2 = torch.Tensor(inputs_a.size()).cuda(), torch.Tensor(inputs_a.size()).cuda()
            y_a2, y_b2 = torch.Tensor(y_a.size()).cuda(), torch.Tensor(y_a.size()).cuda()

            # Select proper points
            x_a2[left_idx], x_b2[left_idx] = inputs_lam[left_idx], x_b[left_idx]
            x_a2[right_idx], x_b2[right_idx] = x_a[right_idx], inputs_lam[right_idx]
            y_a2[left_idx], y_b2[left_idx] = y_lam[left_idx], y_b[left_idx]
            y_a2[right_idx], y_b2[right_idx] = y_a[right_idx], y_lam[right_idx]

            # Update
            x_a, x_b, y_a, y_b = x_a2, x_b2, y_a2, y_b2

            inputs_lam = linear_solv(x_a, x_b, y_a, y_b, inputs_a, inputs_b)
            outputs_lam, _ = net(inputs_lam)
            prob_lam = F.softmax(outputs_lam, dim=1).data
            y_lam = torch.masked_select(prob_lam, true_onehot) - torch.masked_select(prob_lam, shuffle_onehot)

        # Handling the numerical issues
        lamdas = torch.clamp((inputs_lam - inputs_a) / (inputs_b - inputs_a + eps), min=0, max=1)
        lam = torch.ones(num_correct).cuda()
        for i in range(num_correct):
            select_idx = lamdas[i].gt(0) * lamdas[i].lt(1)
            if torch.sum(select_idx) == 0:
                lam[i] = 1
            else:
                lam[i] = 1 - torch.mean(lamdas[i][select_idx])

        # With multiplication of 1e-5, scale of two losses become almost equal ( fixed )
        dist1 = torch.norm((inputs_lam - inputs_a).reshape(num_correct, -1), 2, 1)
        dist2 = torch.norm((inputs_b - inputs_lam).reshape(num_correct, -1), 2, 1)
        dist = 1e-5 * mse_loss(dist1, dist2)

        lams[pred_idx, 0, 0, 0] = lam

    net.train()

    return lams, dist


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    reg_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        inputs_a, inputs_b, targets_a, targets_b, lam = mixup_data(inputs, targets, args.alpha, use_cuda)
        mixed_inputs = lam * inputs_a + (1 - lam) * inputs_b
        outputs, _ = net(mixed_inputs)
        loss = torch.mean(mixup_criterion(criterion, outputs, targets_a, targets_b, lam))

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)

        correct += ((lam * predicted.eq(targets_a.data).float()).cpu().sum().float()
                    + ((1 - lam) * predicted.eq(targets_b.data).float()).cpu().sum().float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar(batch_idx, len(trainloader),
                     'Loss: %.3f | Reg: %.5f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), reg_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return train_loss/batch_idx, reg_loss/batch_idx, 100.*correct/total


def train_boundary(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    reg_loss = 0
    correct = 0
    total = 0
    total_correct = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        mix_idx, prob, pred = correct_random(inputs, targets)
        num_correct = int(torch.sum(pred.data))

        lam_grad, dist = lam_binary_search(inputs, targets, args.num_search, mix_idx, prob, pred)
        # it does not require grad
        lam = lam_grad.data

        lams = lam.repeat(1, inputs.size(1), inputs.size(2), inputs.size(3))
        inputs_a, inputs_b, targets_a, targets_b = inputs, inputs[mix_idx], targets, targets[mix_idx]

        mixed_inputs = lams * inputs_a + (1 - lams) * inputs_b
        lam = lam.reshape(lam.size(0))
        outputs, _ = net(mixed_inputs)
        loss = torch.mean(mixup_criterion(criterion, outputs, targets_a, targets_b, lam))

        if args.beta > 0:
            margin_loss = args.beta * dist
            loss += margin_loss

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        total_correct += num_correct

        correct += ((lam * predicted.eq(targets_a.data).float()).cpu().sum().float()
                    + ((1 - lam) * predicted.eq(targets_b.data).float()).cpu().sum().float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar(batch_idx, len(trainloader),
                     'Loss: %.3f | Reg: %.5f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), reg_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return train_loss/batch_idx, reg_loss/batch_idx, 100.*correct/total


def val(epoch, test):
    global best_val
    global best_acc
    net.eval()
    val_loss = 0.0
    correct = 0.0
    total = 0.0
    major_correct = 0.0
    major_total = 0.0
    minor_correct = 0.0
    minor_total = 0.0

    class_acc = torch.zeros(num_class)
    class_correct = torch.zeros(num_class)
    class_total = torch.zeros(num_class)

    # Define a data loader for evaluating
    if test:
        loader = testloader
    else:
        loader = valiloader

    for batch_idx, (inputs, targets) in enumerate(loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, requires_grad=False), Variable(targets, requires_grad=False)
        outputs, _ = net(inputs)
        loss = criterion(outputs, targets)

        val_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float()

        minor_idx = targets.lt(args.num_imb).float()
        minor_total += minor_idx.cpu().sum()
        minor_correct += (predicted.eq(targets).float() * minor_idx).cpu().sum()
        major_total = total - minor_total
        major_correct = correct - minor_correct

        for i in range(num_class):
            class_idx = targets.eq(i).float()
            class_total[i] += class_idx.cpu().sum()
            class_correct[i] += (predicted.eq(targets).float() * class_idx).cpu().sum()

        progress_bar(batch_idx, len(loader),
                     'Loss: %.3f | Acc: %.3f%% (%d/%d) | Major_ACC: %.3f%% (%d/%d) | Minor ACC: %.3f%% (%d/%d)'
                     % (val_loss/(batch_idx+1), 100.*correct/total, correct, total, 100.*major_correct/major_total
                        ,major_correct, major_total, 100.*minor_correct/minor_total, minor_correct, minor_total))

    acc = 100.*correct/total

    if test:
        class_acc = 100. * class_correct / class_total
        print("========== Class-wise test performance ( avg : {} ) ==========".format(torch.mean(class_acc)))
        print(class_acc)
        np.save('results/class_acc_' + args.name, class_acc.cpu())

    test_loss, test_acc, major_acc, minor_acc = torch.Tensor([0.0]), torch.Tensor([0.0]), torch.Tensor([0.0]), torch.Tensor([0.0])

    if acc > best_val and test == False:
        test_loss, test_acc, major_acc, minor_acc, _, _, _, _ = val(epoch, True)
        best_val = acc
        checkpoint(acc, epoch)
        best_acc = test_acc

    return (val_loss/batch_idx, 100.*correct/total, 100.*major_correct/major_total, 100.*minor_correct/minor_total,
           test_loss, test_acc, major_acc, minor_acc)


def checkpoint(acc, epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt.t7' + args.name)


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = args.lr
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if not os.path.exists(logname):
    with open(logname, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(['epoch', 'train loss', 'reg loss', 'train acc',
                            'val loss', 'minor val acc', 'major val acc', 'val acc',
                            'test loss', 'minor test acc', 'major test acc', 'test acc'])


for epoch in range(start_epoch, args.epoch):
    if epoch >= args.warm:
        train_loss, reg_loss, train_acc = train_boundary(epoch)
    else:
        train_loss, reg_loss, train_acc = train(epoch)

    val_loss, val_acc, major_val, minor_val, test_loss, test_acc, major_acc, minor_acc= val(epoch, False)

    adjust_learning_rate(optimizer, epoch)

    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([epoch, train_loss.cpu().numpy(), reg_loss, train_acc.cpu().numpy(), val_loss.cpu().numpy(),
                            minor_val.cpu().numpy(), major_val.cpu().numpy(), val_acc.cpu().numpy(),
                            test_loss.cpu().numpy(),
                            minor_acc.cpu().numpy(), major_acc.cpu().numpy(), test_acc.cpu().numpy()])

print("Best Accuracy : {}".format(best_acc))


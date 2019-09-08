'''
Training script for CIFAR-10/100
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function
import copy
import argparse
import os
import shutil
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from invertible_network_1d import ReverseConv2, InvertibleResNet, invertible_resnet
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from dataset_1d import Data1D, X_train, X_test, y_train, y_test, num_classes
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

cm = ListedColormap(['#FF0000', '#0000FF','#00FF00'])
cm_bright = ListedColormap(['#FF0000', '#0000FF','#00FF00'])

# define hyper-parameter for reconstruction loss
code_recon_weight = 0.
image_recon_weight = 0.
planes = 2
num_classes = num_classes
inplanes = 2

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('--dataset',default='mnist',type=str,help='dataset')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[30,60],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint_single', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='./checkpoint_single/model_best.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='invertible_resnet',
                    )
parser.add_argument('--depth', type=int, default=20, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
#assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy

def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data
    trainset = Data1D(split='train')
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    testset = Data1D(split='test')
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=True, num_workers=args.workers)

    # Model
    model = invertible_resnet(depth=args.depth,num_classes=num_classes, inplanes = inplanes, planes = planes)
    #model = ResNet(depth = args.depth, num_classes=num_classes)
    InvConv = ReverseConv2(model.conv_feature)
    #InvConv = ReverseConv(depth=args.depth,num_classes=num_classes, inplanes = inplanes, planes = planes)
    #InvConv = torch.nn.DataParallel(InvConv).cuda()
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    # Resume
    title = 'cifar-10-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])



    fig =plt.figure()
    #ax1 = fig.add_subplot(111)

    ind = np.where(y_train==0)[0]
    plt.scatter(X_train[ind, 0], X_train[ind, 1], c=(1,0,0),cmap = cm,
               edgecolors='k', label = 'Class 1', alpha=0.3)

    ind = np.where(y_train == 1)[0]
    plt.scatter(X_train[ind, 0], X_train[ind, 1], c=(0, 1, 0), cmap=cm,
                edgecolors='k', label='Class 2', alpha=0.3)

    inputs1, features1 = generate_boundary(model, 0, 1)
    bound_input = np.concatenate(inputs1, 0)
    plt.scatter(bound_input[:, 0], bound_input[:, 1], c=(0, 1, 1), cmap=cm,
                label='bound_inputary between class 1 and 2')

    if num_classes == 3:
        ind = np.where(y_train == 2)[0]
        plt.scatter(X_train[ind, 0], X_train[ind, 1], c=(0, 0, 1), cmap=cm,
                    edgecolors='k', label='Class 3', alpha=0.3)


    if num_classes == 3:
        inputs2, features2 = generate_boundary(model, 0, 2)
        bound_input = np.concatenate(inputs2, 0)
        plt.scatter(bound_input[:, 0], bound_input[:, 1], c=(1, 0, 1), cmap=cm, label='Boundary between class 1 and 3')


        inputs3, features3 = generate_boundary(model, 1, 2)
        bound_input = np.concatenate(inputs3, 0)
        plt.scatter(bound_input[:, 0], bound_input[:, 1], c=(1, 1, 0), cmap=cm, label='Boundary between class 2 and 3')

    plt.legend(prop={'size': 12})
    plt.title('Bounrary in input domain')
    plt.show()
    # -------------------------------------------------------------------------------------------------------
    # make another plot, display boundaries in feature domain
    # -------------------------------------------------------------------------------------------------------
    plt.figure()
    # -------------------------------------------
    # display data points in feature domain
    colors = [(1,0,0), (0,1,0),(0,0,1)]
    X_train_tensor = torch.from_numpy(X_train).float().cuda()
    feat_train = model.module.conv_feature(X_train_tensor)
    feat_train = feat_train.data.cpu().numpy()

    ind = np.where(y_train == 0)[0]
    plt.scatter(feat_train[ind, 0], feat_train[ind, 1], c=(1, 0, 0), cmap=cm,
                edgecolors='k', label='Class 1', alpha=0.3)

    ind = np.where(y_train == 1)[0]
    plt.scatter(feat_train[ind, 0], feat_train[ind, 1], c=(0, 1, 0), cmap=cm,
                edgecolors='k', label='Class 2', alpha=0.3)

    if num_classes == 3:
        ind = np.where(y_train == 2)[0]
        plt.scatter(feat_train[ind, 0], feat_train[ind, 1], c=(0, 0, 1), cmap=cm,
                    edgecolors='k', label='Class 3', alpha=0.3)

    # -------------------------------------------
    # display boundaries
    bound_feature = np.concatenate(features1,0)
    plt.scatter(bound_feature[:,0],bound_feature[:,1],c=(0,1,1),cmap=cm,label ='Boundary between class 1 and 2')

    if num_classes == 3:
        bound_feature = np.concatenate(features2, 0)
        plt.scatter(bound_feature[:, 0], bound_feature[:, 1], c=(1, 0, 1), cmap=cm, label='Boundary between class 1 and 3')

        bound_feature = np.concatenate(features3, 0)
        plt.scatter(bound_feature[:, 0], bound_feature[:, 1], c=(1, 1, 0), cmap=cm, label='Boundary between class 2 and 3')

    plt.legend(prop={'size': 12})

    plt.title('Boundary in feature domain')
    plt.show()

    # -------------------------------------------------------------------------------------------------------
    # for a single point, display its projection onto the boundary
    # -------------------------------------------------------------------------------------------------------
    # get weights for linear classifier
    inputs = torch.from_numpy(X_train).float().cuda()

    weights = model.module.fc.weight
    biases = model.module.fc.bias

    weight1 = weights[0, ...]
    weight2 = weights[1, ...]

    bias1 = biases[0]
    bias2 = biases[1]

    # generate decision boundary
    # <v,weight1> + bias1 = <v,weight2> + bias2
    # <v, weight1 - weight2> = bias2 - bias1
    #
    # <v_parallel, weight1 - weight2> = bias2 - bias1,  <v_parallel, w> * norm = bias2 -bias1
    # <v_vertical, w> = 0
    # v_vertical = v - <v,w>w if w has norm 1

    feature = model.module.conv_feature(inputs)
    N, C = feature.size()

    w = weight1 - weight2
    norm = torch.sqrt(torch.sum(w ** 2))
    w = w / norm  # normalize w
    w2 = torch.unsqueeze(w,-1)

    # calculate projection of data
    normal = w
    Length = norm
    projection = feature - torch.ger(torch.mm(feature,w2).squeeze(),w)  + normal * (bias2 - bias1) / Length

    # check if is on the decision boundary
    decision_projection = model.module.fc(projection)
    dif = decision_projection[:, 0] - decision_projection[:, 1]
    print(torch.sum(dif ** 2))

    # invert to input space
    input_projection = model.module.conv_feature.inverse(projection)

    # check if inversion is accurate
    decision_projection = model(input_projection)
    dif = decision_projection[:, 0] - decision_projection[:, 1]
    print('Logit difference after inversion is %.7f' % torch.sum(dif ** 2).item())

    # --------------------------------------------------------------------------------------------------
    # plot boundary, data point, and its projection onto the boundary
    input_projection_np = input_projection.data.cpu().numpy()
    feature_projection_np = projection.data.cpu().numpy()
    feature_np = feature.data.cpu().numpy()

    # input domain
    plt.figure()
    plt.scatter(bound_input[:, 0], bound_input[:, 1], c=(0, 1, 1), cmap=cm, label='Boundary in input domain between class 1 and 2')

    plt.scatter(X_train[1, 0], X_train[1, 1], c=(1, 0, 0), cmap=cm,
                edgecolors='k', label='Class 1', alpha=0.3)
    plt.scatter(input_projection_np[1, 0], input_projection_np[1, 1], c=(0, 1, 0), cmap=cm,
                edgecolors='k', label='Class 1', alpha=0.3)
    plt.title('projection in input domain')
    plt.show()
    # feature domain
    plt.figure()
    plt.scatter(bound_feature[:, 0], bound_feature[:, 1], c=(0, 1, 1), cmap=cm, label='Boundary in feature domain between class 1 and 2')

    plt.scatter(feature_np[1, 0], feature_np[1, 1], c=(1, 0, 0), cmap=cm,
                edgecolors='k', label='Class 1', alpha=0.3)
    plt.scatter(feature_projection_np[1, 0], feature_projection_np[1, 1], c=(0, 1, 0), cmap=cm,
                edgecolors='k', label='Class 1', alpha=0.3)
    plt.title('projection in feature domain')
    plt.show()

    return



def generate_boundary(model, ind1 = 0, ind2 = 1):
    # extract vector for two classes
    weights = model.module.fc.weight
    weight1 = weights[ind1,:].data.cpu().numpy()
    weight2 = weights[ind2,:].data.cpu().numpy()

    bias1 = model.module.fc.bias[0].data.cpu().numpy()
    bias2 = model.module.fc.bias[1].data.cpu().numpy()

    # generate a sequence of numbers
    x_feature = np.arange(-200,500,1)

    inputs = []
    features = []
    for x in x_feature:
        # [x,y]*weight1 + bias1 = [x,y]*weight2 + bias2
        # x*weight1[0] + y*weight1[1] + bias1 = x*weight2[0] + y*weight2[1] + bias2
        # x*weight1[0] + bias1 - x*weight2[0] - bias2 = y * (weight2[1] - weight1[1])
        x = x / 200.0
        y =( x*weight1[0] + bias1 - x*weight2[0] - bias2 ) / (weight2[1] - weight1[1])

        feature = np.array([x,y])
        feature = torch.from_numpy(feature).float().cuda()
        feature = torch.unsqueeze(feature,0)

        #reverse to input domain
        input = model.module.inverse(feature)
        input = input.data.cpu().numpy()

        inputs.append(input)
        features.append(feature)

    return inputs, features


if __name__ == '__main__':
    main()

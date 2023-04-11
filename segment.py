from __future__ import print_function
import argparse
import os
from collections import OrderedDict
import shutil
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import math
import models.clsd.builder as simsiam
import utils.semantic_seg as transform
import torch.nn.functional as F
from lib import transforms_for_rot, transforms_back_rot, transforms_for_noise, transforms_for_scale, transforms_back_scale, postprocess_scale
import cv2
import matplotlib.pyplot as plt
from mean_teacher import losses,ramps
from utils import  mkdir_p
from tensorboardX import SummaryWriter
from utils.utils import *
from dataset.imageInput import get_imgInput_test


parser = argparse.ArgumentParser(description='PyTorch MixMatch Training')
# Optimization options
parser.add_argument('--epochs', default=1024, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=64, type=int, metavar='N',
                    help='train batchsize')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pre_train', default='', type=str, metavar='PATH',
                    help='path to pre-train checkpoint (default: none)')
# Miscs
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
# Device options
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
# Method options
parser.add_argument('--n-labeled', type=int, default=250,
                    help='Number of labeled data')
parser.add_argument('--val-iteration', type=int, default=1024,
                    help='Number of labeled data')
parser.add_argument('--data', default='',
                    help='input data path')
parser.add_argument('--out', default='result',
                    help='Directory to output the result')
parser.add_argument('--alpha', default=0.75, type=float)
parser.add_argument('--lambda-u', default=75, type=float)
parser.add_argument('--T', default=0.5, type=float)
parser.add_argument('--ema-decay', default=0.999, type=float)
parser.add_argument('--num-class', default=10, type=int)
parser.add_argument('--evaluate', action="store_true")
parser.add_argument('--wlabeled', action="store_true")
parser.add_argument('--scale', action="store_true")
parser.add_argument('--presdo', action="store_true")
parser.add_argument('--tcsm', action="store_true")
parser.add_argument('--tcsm2', action="store_true")
parser.add_argument('--autotcsm', action="store_true")
parser.add_argument('--multitcsm', action="store_true")
parser.add_argument('--baseline', action="store_true")
parser.add_argument('--test_mode', action="store_true")
parser.add_argument('--test', action="store_true")
parser.add_argument('--output', default='result',
                    help='Directory to output the img')
parser.add_argument('--size', default=224, type=int,
                    help='the size of the input')
parser.add_argument('--test_post', action="store_true")

# lr
parser.add_argument("--lr_mode", default="cosine", type=str)
parser.add_argument("--lr", default=0.03, type=float)
parser.add_argument("--warmup_epochs", default=0, type=int)
parser.add_argument("--warmup_lr", default=0.0, type=float)
parser.add_argument("--targetlr", default=0.0, type=float)

#
parser.add_argument('--consistency_type', type=str, default="mse")
parser.add_argument('--consistency', type=float,  default=10.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=400.0, help='consistency_rampup')

#
parser.add_argument('--initial-lr', default=0.0, type=float,
                    metavar='LR', help='initial learning rate when using linear rampup')
parser.add_argument('--lr-rampup', default=0, type=int, metavar='EPOCHS',
                    help='length of learning rate rampup in the beginning')
parser.add_argument('--lr-rampdown-epochs', default=None, type=int, metavar='EPOCHS',
                    help='length of learning rate cosine rampdown (>= length of training)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}



# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

# cudnn.enabled = True
# cudnn.benchmark=True
# cudnn.deterministic = True

#强制同步
os.environ['CUDA_LAUNCH_BLOCKING']='1'
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
np.random.seed(args.manualSeed)

best_acc = 0  # best test accuracy
NUM_CLASS = args.num_class

from shutil import copyfile


class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, input, targets):

        N = targets.size()[0]

        smooth = 1

        input_flat = input.view(N, -1)
        targets_flat = targets.view(N, -1)


        intersection = input_flat * targets_flat
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)

        loss = 1 - N_dice_eff.sum() / N
        return loss



def main():
    global best_acc

    if not os.path.isdir(args.out):
        mkdir_p(args.out)
    copyfile("train_tcsm_mean.py", args.out+"/train_tcsm_mean.py")


    if args.retina:
        mean = [22, 47, 82]
    else:
        mean = [140,150,180]
    std = None

    # Data augmentation
    # print(f'==> Preparing imgInput dataset')
    transform_train = transform.Compose([
        transform.RandomRotationScale(args.size),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])

    transform_val = transform.Compose([
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])

    
    import dataset.imageInput as dataset
    train_labeled_set, train_unlabeled_set, val_set, test_set = dataset.get_imgInput_dataset("./data/img/",
                                                num_labels=args.n_labeled,
                                                transform_train=transform_train,
                                                transform_val=transform_val,
                                                 transform_forsemi=None)

    labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True,
                                          num_workers=2, drop_last=True)
    # print(type(labeled_trainloader))
    # print(labeled_trainloader)

    if args.baseline:
        unlabeled_trainloader = None
    else:
        unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=args.batch_size, shuffle=True,
                                            num_workers=2, drop_last=True)

    val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)



    # Model
    print("==> creating model")

    def create_model(ema=False):
        model = simsiam.ConNet(base_encoder=simsiam.DenseUnet_2d_ce(),base_classifier=simsiam.Classifier())
        model = model.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()

        return model

    model = create_model()
    ema_model = create_model(ema=True)

    # freeze all layers but the last fc
    # for name, param in model.named_parameters():
    #     if name not in ['fc.weight', 'fc.bias']:
    #         param.requires_grad = False         #除了最后的fc，其他层不训练

    # init the fc layer
    # model.fc.weight.data.normal_(mean=0.0, std=0.01)
    # model.fc.bias.data.zero_()



    # cudnn.benchmark = True

    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.93, 8.06]).cuda())
    # criterion = BinaryDiceLoss().cuda()
    # criterion = nn.BCELoss(weight=torch.FloatTensor([1.93, 8.06])).cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    init_lr = args.lr * args.batch_size / args.batch_size  # lr = 0.5
    start_epoch = 0

    # Resume
    if args.resume:
        print('==> Resuming from checkpoint..' + args.resume)
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.resume)
        print("epoch ", checkpoint['epoch'])
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    # Pre-train
    if args.pre_train:
        if os.path.isfile(args.pre_train):
            print("=> loading checkpoint '{}'".format(args.pre_train))
            checkpoint = torch.load(args.pre_train, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']

            for k in list(state_dict.keys()):
                # retain only encoder up to before the embedding layer
                if k.startswith('predictor') or k.startswith('projector'):
                    del state_dict[k]

            args.start_epoch = 0
            model.load_state_dict(state_dict, strict=False)

            print("=> loaded pre-trained model '{}'".format(args.pre_train))
        else:
            print("=> no checkpoint found at '{}'".format(args.pre_train))

    if args.evaluate:  # 评价
        val_loss, val_result = multi_validate(val_loader, ema_model, criterion, 0, use_cuda, args)
        print("val_loss", val_loss)
        print("Val ema_model : JA, AC, DI, SE, SP \n")
        print(", ".join("%.4f" % f for f in val_result))

        val_loss, val_result = multi_validate(val_loader, model, criterion, 0, use_cuda, args)
        print("val_loss", val_loss)
        print("Val model: JA, AC, DI, SE, SP \n")
        print(", ".join("%.4f" % f for f in val_result))
        return

    if args.test:  
        if not os.path.isdir(args.output):
            mkdir_p(args.output)
        if args.baseline:
            function = model
        else:
            # function=ema_model
            function = model  
        function.eval()
        with torch.no_grad():
            num = 1
            for batch_idx, (inputs, targets, name) in enumerate(test_loader):
                if use_cuda:
                    targets = targets.long()
                    targets[targets == 255] = 1
                    inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)

                # compute output
                outputs = function(inputs)[0]
                outputs = F.softmax(outputs, dim=1)
                x = outputs.cpu().detach().numpy()
                z = outputs.cpu().detach().numpy()
                y = targets.cpu().detach().numpy()
                results = post_process_evaluate(x, y, name, args)

                # 输出图片
                # print("x_shape:",x.shape)#(batch, 2, 1360, 1360)

                z = (z * 255).astype(np.uint8)

                for j in range(x.shape[0]):
                    img = z[j][1]
                    cv2.imwrite(os.path.join(args.output, str(num).zfill(4) + '.png'), img)
                    print("Done:" + str(num).zfill(4) + '.png')
                    num += 1

        return

    total_acc = []
    epochs = []
    LOSS = [10000]
    for epoch in range(start_epoch, args.epochs):

        # train
        loss = train_meanteacher(labeled_trainloader, unlabeled_trainloader, model, ema_model, optimizer, criterion,
                                 epoch)
        if loss < min(LOSS):
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best=True, filename='checkpoint_{:03d}_{:.6f}.pth.tar'.format(epoch, loss))
        LOSS.append(loss)
        total_acc.append(loss)
        epochs.append(epoch)

    print(epochs)
    print(total_acc)
    plt.plot(epochs, total_acc)
    plt.savefig('acc.png')


def train_meanteacher(labeled_trainloader, unlabeled_trainloader, model, ema_model, optimizer,
                    criterion, epoch):
    global global_step

    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    # switch to train mode
    model.train()
    ema_model.train()
    LOSS = 0
    for batch_idx in range(args.val_iteration):
        print("epoch:{} batch_id:{}".format(epoch, batch_idx))
        try:
            inputs_x, targets_x, name_x = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x, name_x = labeled_train_iter.next()

        # print(targets_x.shape)# 5 224 224
        # print(inputs_x.shape)
        inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)

        if not args.baseline:
            try:
                inputs_u, inputs_u2 = unlabeled_train_iter.next()
            except:
                unlabeled_train_iter = iter(unlabeled_trainloader)
                inputs_u, inputs_u2 = unlabeled_train_iter.next()

            if use_cuda:
                # targets_x[targets_x == 255] = 1
                inputs_u = inputs_u.cuda()
                inputs_u2 = inputs_u2.cuda()

            with torch.no_grad():


                # tcsm
                inputs_u2_noise = transforms_for_noise(inputs_u2, 0.5)

                inputs_u2_noise, rot_mask, flip_mask = transforms_for_rot(inputs_u2_noise)

                # add scale
                if args.scale:
                    inputs_u2_noise, scale_mask = transforms_for_scale(inputs_u2_noise, args.size)


                outputs = model(inputs_u)
                outputs_ema = ema_model(inputs_u2_noise)

                outputs_u = outputs[0]
                outputs_u_ema = outputs_ema[0]


                if args.scale:
                    outputs_u_ema, scale_mask = transforms_back_scale(outputs_u_ema, scale_mask, args.size)
                    outputs_u = postprocess_scale(outputs_u, scale_mask, args.size)

                # tcsm back: modify ema output
                outputs_u_ema = transforms_back_rot(outputs_u_ema, rot_mask, flip_mask)




        # iter_num
        iter_num = batch_idx + epoch * args.val_iteration
        # lr = adjust_learning_rate(optimizer, epoch, batch_idx, args.val_iteration)

        # labeled data
        logits_x = model(inputs_x)[0]

        outputs_soft = F.softmax(logits_x, dim=1)
        # print(outputs_soft)
        # print("================================================")
        index = torch.where(targets_x>0)
        targets_x[index]=1.0
        # print(targets_x)

        # Lx = criterion(outputs_soft,targets_x.long())
        #  = dice_loss()
        mse = nn.MSELoss().cuda()
        dice_loss=BinaryDiceLoss().cuda()
        Lx_dice=dice_loss(outputs_soft[:, 1, :, :], targets_x.long())

        # Lx = 0.5 * (Lx + Lx_dice)
        Lx = Lx_dice
        # unlabeled data
        if not args.baseline:
            
            consistency_weight = get_current_consistency_weight(epoch)

            consistency_dist = consistency_criterion(outputs_u, outputs_u_ema)
            consistency_dist = torch.mean(consistency_dist)

            Lu = consistency_weight * consistency_dist
           
            p1,p2,z1,z2 = outputs[1],outputs_ema[1],outputs[2],outputs_ema[2]
            Lc = (mse(p1, z2).mean() + mse(p2, z1).mean()) * 0.5

            loss = Lx + Lu + Lc
            # loss = Lx + Lu
        else:
            loss = Lx

        loss.requires_grad_(True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        update_ema_variables(model, ema_model, args.ema_decay, iter_num)

        LOSS = loss.cpu().detach().numpy()

    return LOSS




def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr



def save_checkpoint(state, is_best, checkpoint=args.out, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))









if __name__ == '__main__':
    start=time.time()
    main()
    end = time.time()
    print(end-start)

import os
import argparse
import time
from tqdm import tqdm
import numpy as np

import torch
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import shutil
import timm.scheduler as timm_scheduler

import loaders 
import models
from metrics import ildv
import math

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', default='', type=str, help='model name')
    parser.add_argument('--data_name', default='', type=str, help='data name')
    parser.add_argument('--num_classes', default='', type=int, help='num classes')
    parser.add_argument('--num_epochs', default=200, type=int, help='num epochs')
    parser.add_argument('--batch_size', default=512, type=int, help='batch size')
    parser.add_argument('--model_dir', default='', type=str, help='model dir')
    parser.add_argument('--result_name', default='', type=str, help='result name')
    parser.add_argument('--train_data_dir', default='', type=str, help='train_data_dir')
    parser.add_argument('--test_data_dir', default='', type=str, help='test_data_dir')
    parser.add_argument('--log_dir', default='', type=str, help='log dir')
    args = parser.parse_args()

    # ----------------------------------------
    # basic configuration
    # ----------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dir = args.train_data_dir
    test_dir = args.test_data_dir

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if os.path.exists(args.log_dir):
        shutil.rmtree(args.log_dir)

    print('-' * 50)
    print('TRAIN ON:', device)
    print('TRAIN DATA DIR:', args.train_data_dir)
    print('TEST DATA DIR:', args.test_data_dir)
    print('MODEL DIR:', args.model_dir)
    print('LOG DIR:', args.log_dir)
    print('RESULT_NAME:', args.result_name + '.pth')
    print('-' * 50)

    # ----------------------------------------
    # trainer configuration
    # ----------------------------------------
    model = models.load_model(args.model_name, args.data_name, num_classes=args.num_classes)
    # model.load_state_dict(torch.load('/mnt/nfs/hjc/project/td/output/train_models/pth/imagenet/deit_imagenet_baseline.pth'), strict=False)
    model.to(device)
    print('MODEL LOAD DONE')
    train_loader = loaders.load_data(train_dir, args.data_name, data_type='train', batch_size=args.batch_size)
    print('TRAIN LOAD DONE')
    test_loader = loaders.load_data(test_dir, args.data_name, data_type='test', batch_size=args.batch_size)
    print('TEST LOAD DONE')
    modules = models.load_modules(model=model)

    criterion = nn.CrossEntropyLoss()

    batch_size = args.batch_size
    max_learn_rate = 0.0005 * (batch_size / 512)
    optimizer = optim.AdamW(params=model.parameters(), lr=max_learn_rate, weight_decay=0.05, eps=1e-8)
    num_steps = int(args.num_epochs * len(train_loader))
    warmup_steps = 0
    warmup_lr = 1e-06
    scheduler = timm_scheduler.CosineLRScheduler(
        optimizer,
        t_initial=(num_steps - warmup_steps),
        lr_min=1e-05,
        warmup_lr_init=warmup_lr,
        warmup_t=warmup_steps,
        cycle_limit=1,
        t_in_epochs=False,
    )

    # optimizer = optim.SGD(params=model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-2)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.num_epochs)

    writer = SummaryWriter(args.log_dir)

    # ----------------------------------------
    # each epoch
    # ----------------------------------------
    since = time.time()

    best_acc = None
    best_epoch = None

    for epoch in tqdm(range(args.num_epochs)):
        acc1, loss = train(train_loader, model, criterion, optimizer, device, args)
        writer.add_scalar(tag='training acc1', scalar_value=acc1, global_step=epoch)
        writer.add_scalar(tag='training loss', scalar_value=loss, global_step=epoch)
        acc1, loss = test(test_loader, model, criterion, device, args)
        writer.add_scalar(tag='test acc1', scalar_value=acc1, global_step=epoch)
        writer.add_scalar(tag='test loss', scalar_value=loss, global_step=epoch)

        # ----------------------------------------
        # save best model
        # ----------------------------------------
        if best_acc is None or best_acc < acc1:
            best_acc = acc1
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(args.model_dir, args.result_name + '.pth'))
        print(best_acc)

        scheduler.step(epoch)

    print('COMPLETE !!!')
    print('BEST ACC', best_acc)
    print('BEST EPOCH', best_epoch)
    print('TIME CONSUMED', time.time() - since)


def train(train_loader, model, criterion, optimizer, device, args):

    model.train()

    acc1 = ildv.MulticlassAccuracy(average='macro', num_classes=args.num_classes).to(device)

    print('LEN OF TRAIN_LOADER')
    print(len(train_loader))
    for samples in tqdm(enumerate(train_loader)):
        _, tmp = samples
        inputs, labels, names = tmp
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        acc1.update(outputs, labels)
        optimizer.zero_grad()  # 1
        loss.backward()  # 2
        optimizer.step()  # 3
    
    return acc1.compute().item(), loss.item()


def test(test_loader, model, criterion, device, args):

    model.eval()

    acc1 = ildv.MulticlassAccuracy(average='macro', num_classes=args.num_classes).to(device)

    for samples in enumerate(test_loader):
        _, tmp = samples
        inputs, labels, names = tmp
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        acc1.update(outputs, labels)
        
    return acc1.compute().item(), loss.item()


if __name__ == '__main__':
    main()

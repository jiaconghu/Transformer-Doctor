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
from utils.misc import reduce_dict, update, MetricLogger, SmoothedValue
from evaluator.default import DefaultEvaluator
from criterions import cross_entropy

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', default='', type=str, help='model name')
    parser.add_argument('--data_name', default='', type=str, help='data name')
    parser.add_argument('--num_classes', default='', type=int, help='num classes')
    parser.add_argument('--num_epochs', default=200, type=int, help='num epochs')
    parser.add_argument('--model_dir', default='', type=str, help='model dir')
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
    print('-' * 50)

    # ----------------------------------------
    # trainer configuration
    # ----------------------------------------
    model = models.load_model(args.model_name, args.data_name, num_classes=args.num_classes)
    model.to(device)
    print('MODEL LOAD DONE')
    train_loader = loaders.load_data(train_dir, args.data_name, data_type='train')
    print('TRAIN LOAD DONE')
    test_loader = loaders.load_data(test_dir, args.data_name, data_type='test')
    print('TEST LOAD DONE')

    # criterion = nn.CrossEntropyLoss()
    criterion = cross_entropy.CrossEntropy(losses=['labels'], weight_dict={'loss_ce': 1})
    print(criterion)

    batch_size = 256
    learn_rate = 0.0005 * (batch_size / 512)
    optimizer = optim.AdamW(params=model.parameters(), lr=learn_rate, weight_decay=0.05, eps=1e-8)
    num_steps = int(args.num_epochs * len(train_loader))
    warmup_epochs = 5
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
            torch.save(model.state_dict(), os.path.join(args.model_dir, 'tvit_imagenet10_baseline_test.pth'))
        print(best_acc)

        scheduler.step(epoch)

    print('COMPLETE !!!')
    print('BEST ACC', best_acc)
    print('BEST EPOCH', best_epoch)
    print('TIME CONSUMED', time.time() - since)


def train(train_loader, model, criterion, optimizer, device, args):

    model.train()
    criterion.train()

    evaluator = DefaultEvaluator(metrics=['acc', 'recall', 'precision', 'f1'])

    print('LEN OF TRAIN_LOADER')
    print(len(train_loader))
    for samples in tqdm(enumerate(train_loader)):
        _, tmp = samples
        inputs, labels, names = tmp
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)

        # loss = criterion(outputs, labels)
        loss_dict = criterion(outputs, labels)
        print(loss_dict)
        weight_dict = criterion.weight_dict
        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        evaluator.update(outputs, labels)

        optimizer.zero_grad()  # 1
        # if scaler:
        #     scaler.scale(loss).backward()
        #     if max_norm > 0:
        #         scaler.unscale_(optimizer)
        #         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        #     scaler.step(optimizer)
        #     scaler.update()
        loss.backward()  # 2
        optimizer.step()  # 3

    acc = evaluator.metric_acc(evaluator.outputs, evaluator.targets)
    
    return acc, loss.item()


def test(test_loader, model, criterion, device, args):

    model.eval()

    evaluator = DefaultEvaluator(metrics=['acc', 'recall', 'precision', 'f1'])

    for samples in enumerate(test_loader):
        _, tmp = samples
        inputs, labels, names = tmp
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        evaluator.update(outputs, labels)
    
    acc = evaluator.metric_acc(evaluator.outputs, evaluator.targets)
        
    return acc, loss.item()


if __name__ == '__main__':
    main()

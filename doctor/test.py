import argparse
import os.path
import re
import time
from collections import OrderedDict

import numpy as np
from tqdm import tqdm

import torch
from torch import nn

import loaders
import models
from metrics import ildv

from utils.misc import reduce_dict, update, MetricLogger, SmoothedValue
from evaluator.default import DefaultEvaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='', type=str, help='model name')
    parser.add_argument('--num_classes', default='', type=int, help='num classes')
    parser.add_argument('--model_path', default='', type=str, help='model path')
    parser.add_argument('--data_name', default='', type=str, help='data name')
    parser.add_argument('--data_dir', default='', type=str, help='data directory')
    parser.add_argument('--save_dir', default='', type=str, help='save directory')
    args = parser.parse_args()

    # ----------------------------------------
    # basic configuration
    # ----------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    print('-' * 50)
    print('DEVICE:', device)
    print('MODEL PATH:', args.model_path)
    print('DATA PATH:', args.data_dir)
    print('-' * 50)

    # ----------------------------------------
    # test configuration
    # ----------------------------------------
    model = models.load_model(model_name=args.model_name, data_name=args.data_name, num_classes=args.num_classes)
    # model.load_state_dict(torch.load(args.model_path)['model'],strict=False)
    print(model.load_state_dict(torch.load(args.model_path), strict=False))

    model.to(device)

    test_loader = loaders.load_data(data_dir=args.data_dir, data_name=args.data_name, data_type='test', batch_size=512)

    # 计算方法
    # evaluates = {
    #     'A': [ildv.MulticlassAccuracy(average='macro', num_classes=args.num_classes).to(device),
    #           ildv.MulticlassF1Score(average='macro', num_classes=args.num_classes).to(device)],
    #     'B': [ildv.MulticlassFalseNegativeRate(average='macro', num_classes=args.num_classes).to(device),
    #           ildv.MulticlassFalseDiscoveryRate(average='macro', num_classes=args.num_classes).to(device)],
    #     'C': [ildv.MulticlassBalancedAccuracy(args.num_classes).to(device)]
    # }
    evaluates = DefaultEvaluator(metrics=['acc', 'recall', 'precision', 'f1'])

    # ----------------------------------------
    # each epoch
    # ----------------------------------------
    since = time.time()

    scores = test(test_loader, model, evaluates, device)
    save_path = os.path.join(args.save_dir, '{}_{}.npy'.format(args.model_name, args.data_name))
    np.save(save_path, scores)

    print('-' * 50)
    print('TIME CONSUMED', time.time() - since)


# def test(test_loader, model, evaluates, device):
#     model.eval()

#     for i, samples in enumerate(tqdm(test_loader)):
#         # print(samples)
#         tmp = samples
#         inputs, labels, names = tmp
#         inputs = inputs.to(device)
#         labels = labels.to(device)

#         with torch.no_grad():
#             outputs = model(inputs)

#         for value in evaluates.values():
#             for evaluate in value:
#                 evaluate.update(outputs, labels)

#     # calculate result
#     scores = {}
#     for key, value in zip(evaluates.keys(), evaluates.values()):
#         scores[key] = []
#         for evaluate in value:
#             score = evaluate.compute().cpu().item()
#             scores[key].append(score)


#     print(scores)

#     return scores

def test(test_loader, model, evaluates, device):
    model.eval()

    for i, samples in enumerate(tqdm(test_loader)):
        # print(samples)
        tmp = samples
        inputs, labels, names = tmp
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.cuda.amp.autocast(enabled=False):
            outputs = model(inputs)

        evaluates.update(outputs, labels)
    print('ACC:', evaluates.metric_acc(evaluates.outputs, evaluates.targets))

    return evaluates


if __name__ == '__main__':
    main()

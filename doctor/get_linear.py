import sys

sys.path.append('.')

import os
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn

import models
import loaders
import pickle


class LinearProcess:
    def __init__(self):
        self.linears = [[] for _ in range(10)]
        self.linear_names = [[] for _ in range(10)]

    def __call__(self, linears, labels, names):
        for i, linear in enumerate(linears):
            label = labels[i]
            name = names[i]
            self.linears[label].append(linear)
            self.linear_names[label].append(name)

    def save(self):
        data_path = '/nfs3-p1/ch/project/td/output/vit_high_cifar10/train_images'
        class_names = sorted([d.name for d in os.scandir(data_path) if d.is_dir()])
        for i in range(10):
            linears = self.linears[i]
            result = np.empty([0,0])
            for j, linear in enumerate(linears):
                linear = linear.cpu().numpy()
                for k in range(64):
                    if (k == 0):
                        result = linear[k + 1][:50]
                    else:                        
                        result = np.vstack([result, linear[k + 1][:50]])
                class_name = class_names[i]
                sample_name = self.linear_names[i][j]
                original_path = '/nfs3-p1/ch/project/td/output/vit_high_cifar10/linears'
                save_path = os.path.join(original_path, class_name)
                file_path = os.path.join(save_path, sample_name)
                print(file_path)
                print(result.shape)
                np.save(file_path, result)
            break

            # 把result可视化


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', default='', type=str, help='model name')
    parser.add_argument('--data_name', default='', type=str, help='data name')
    parser.add_argument('--num_classes', default='', type=int, help='num classes')
    parser.add_argument('--model_path', default='', type=str, help='model path')
    parser.add_argument('--data_dir', default='', type=str, help='data path')
    parser.add_argument('--save_dir', default='', type=str, help='locating path')
    args = parser.parse_args()

    # ----------------------------------------
    # basic configuration
    # ----------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    print('-' * 50)
    print('TRAIN ON:', device)
    print('DATA DIR:', args.data_dir)
    print('SAVE DIR:', args.save_dir)
    print('-' * 50)

    # ----------------------------------------
    # model/data configuration
    # ----------------------------------------
    model = models.load_model(model_name=args.model_name, data_name=args.data_name, num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    model.eval()

    data_loader = loaders.load_data(args.data_dir, args.data_name, data_type='train')

    my_linear = LinearProcess()


    # ----------------------------------------
    # forward
    # ----------------------------------------
    for i, samples in enumerate(tqdm(data_loader)):
        inputs, labels, names = samples
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs, linears = model(inputs)
            my_linear(linears, labels, names)
    my_linear.save()

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    main()
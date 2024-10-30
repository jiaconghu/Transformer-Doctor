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


class HookModule:
    def __init__(self, module):
        self.module = module
        self.inputs = None
        self.outputs = None
        module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, outputs):
        self.inputs = inputs[0]
        self.outputs = outputs


class ComponentLocating:
    def __init__(self, modules, num_classes):
        self.modules = [HookModule(module) for module in modules]
        self.values = [[[] for _ in range(num_classes)] for _ in range(len(modules))]
        self.num_classes = num_classes

    def __call__(self, outputs, labels):
        for layer in range(12):
            values = self.modules[int(4 * layer + 3)].outputs
            values = torch.relu(values)
            values = values.detach().cpu().numpy()
            for b in range(len(labels)):
                self.values[layer][labels[b]].append(values[b])
            print('layer: ', layer, '   shape: ', np.shape(self.values[layer][0]))


    def sift(self, result_path):
        for layer in range(12):
            for label in range(100):
                values = self.values[layer][label]     # (num_images, channels)
                save_path = os.path.join(result_path, 'layer_{}_label_{}.npy'.format(layer, label))
                np.save(save_path, values)
                print(save_path)
            




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
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

    mask_dir = '/mnt/nfs/hjc/project/td/output/ideal/atts/imagenet'
    data_loader = loaders.load_data_mask(args.data_dir, args.data_name, mask_dir, data_type='test')

    modules = models.load_modules(model=model)

    component_locating = ComponentLocating(modules=modules, num_classes=args.num_classes)

    # ----------------------------------------
    # forward
    # ----------------------------------------
    for i, samples in enumerate(tqdm(data_loader)):
        inputs, labels, names, masks = samples
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        component_locating(outputs, labels)

    component_locating.sift(result_path=args.save_dir)


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    main()

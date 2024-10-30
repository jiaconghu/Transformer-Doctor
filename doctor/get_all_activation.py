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


class HookModule:
    def __init__(self, module):
        self.module = module
        self.inputs = None
        self.outputs = None
        module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, outputs):
        self.inputs = inputs[0]
        self.outputs = outputs

    
class ActivationGet:
    def __init__(self, modules):
        self.modules = [HookModule(module) for module in modules]
        
    def __call__(self, names, outputs, labels):
        nll_loss = torch.nn.NLLLoss()(outputs, labels)

        for layer in range(12):
            activtations = self.modules[int(4 * layer + 3)].outputs        
            activtations = activtations.cpu().detach().numpy()
            folder_path = '/mnt/nfs/hjc/project/td/output/tmp/activations/npys/{}'.format(layer)
            count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
            save_path = os.path.join(folder_path, 'img_{}'.format(count))
            np.save(save_path, activtations)
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

    attention_get = ActivationGet(modules)

    # ----------------------------------------
    # forward
    # ----------------------------------------
    for i, samples in enumerate(tqdm(data_loader)):
        inputs, labels, names, masks = samples
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        attention_get(names, outputs, labels)


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    main()
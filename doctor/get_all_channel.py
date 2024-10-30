"""
activation:
each layer

gradient:
output to each layer
"""
import sys

sys.path.append('.')

import argparse
import torch
import numpy as np
import os
from tqdm import tqdm

import models
import loaders


class HookModule:
    def __init__(self, module):
        self.module = module
        self.inputs = None
        self.outputs = None
        module.register_forward_hook(self._hook)

    def grads(self, outputs, inputs=None, retain_graph=True, create_graph=True):
        if inputs is None:
            inputs = self.outputs  # default the output dim

        return torch.autograd.grad(outputs=outputs,
                                   inputs=inputs,
                                   retain_graph=retain_graph,
                                   create_graph=create_graph)[0]

    def _hook(self, module, inputs, outputs):
        self.inputs = inputs[0]
        self.outputs = outputs


def _normalization(data, axis=None, bot=False):
    assert axis in [None, 0, 1]
    _max = np.max(data, axis=axis)
    if bot:
        _min = np.zeros(_max.shape)
    else:
        _min = np.min(data, axis=axis)
    _range = _max - _min
    if axis == 1:
        _norm = ((data.T - _min) / (_range + 1e-5)).T
    else:
        _norm = (data - _min) / (_range + 1e-5)
    return _norm


class getChannels:
    def __init__(self, modules, num_classes):
        self.modules = [HookModule(module) for module in modules]
        # [num_modules, num_labels, num_images, channels]
        self.grads = [[[] for _ in range(num_classes)] for _ in range(len(modules))]
        self.activations = [[[] for _ in range(num_classes)] for _ in range(len(modules))]

    def __call__(self, outputs, labels):
        # nll_loss = torch.nn.NLLLoss()(outputs, labels)
        for layer, module in enumerate(self.modules):
            # grads = module.grads(-nll_loss, module.outputs)
            # grads = torch.relu(grads)
            # grads = grads.detach().cpu().numpy()
            activations = module.outputs
            activations = torch.relu(activations)
            activations = activations.detach().cpu().numpy()
            for b in range(len(labels)):
                # self.grads[layer][labels[b]].append(grads[b])
                self.activations[layer][labels[b]].append(activations[b])

    def sift(self, result_path, threshold):
        for layer in range(12):
            # grads = self.grads[3 + layer * 5]
            activations = self.activations[3 + layer * 5]
            # grads = np.asarray(grads)
            activations = np.asarray(activations)
            # if len(grads.shape) == 4:
            #     grads = np.squeeze(grads[:, :, 0, :])                   # [num_labels, num_images, channels]
            if len(activations.shape) == 4:
                activations = np.squeeze(activations[:, :, 0, :])       # [num_labels, num_images, channels]

            values = activations

            for label, value in enumerate(values):
                value = _normalization(value, axis=1)       # [num_images, channels]
                
                value_path = os.path.join(result_path, '{}_{}.npy'.format(label, layer))
                np.save(value_path, value)
                print(value_path)

            # mask = np.zeros(value.shape)                # [num_images, channels]
            # mask[np.where(value > threshold)] = 1       # [num_images, channels]
            # mask = np.sum(mask, axis=0)                 # [channels]
            # mask = np.where(mask > 2, 1, 0)             # [channels]

            # mask_path = os.path.join(result_path, '{}.npy'.format(label))
            # np.save(mask_path, mask)
            # print(mask_path)


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', default='', type=str, help='model name')
    parser.add_argument('--data_name', default='', type=str, help='data name')
    parser.add_argument('--in_channels', default='', type=int, help='in channels')
    parser.add_argument('--num_classes', default='', type=int, help='num classes')
    parser.add_argument('--model_path', default='', type=str, help='model path')
    parser.add_argument('--data_path', default='', type=str, help='data path')
    parser.add_argument('--grad_path', default='', type=str, help='grad path')
    parser.add_argument('--theta', default='', type=float, help='theta')
    parser.add_argument('--device_index', default='0', type=str, help='device index')
    args = parser.parse_args()

    # ----------------------------------------
    # basic configuration
    # ----------------------------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_index
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(args.grad_path):
        os.makedirs(args.grad_path)

    print('-' * 50)
    print('TRAIN ON:', device)
    print('DATA PATH:', args.data_path)
    print('RESULT PATH:', args.grad_path)
    print('-' * 50)

    # ----------------------------------------
    # model/data configuration
    # ----------------------------------------
    model = models.load_model(model_name=args.model_name, data_name=args.data_name, num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    model.eval()

    data_loader = loaders.load_data(data_dir=args.data_path, data_name=args.data_name, data_type='test', batch_size=512)

    modules = models.load_modules(model=model)
    print(modules)

    get_channels = getChannels(modules=modules, num_classes=10)

    # ----------------------------------------
    # forward
    # ----------------------------------------
    for i, samples in enumerate(tqdm(data_loader)):
        inputs, labels, names = samples
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)

        get_channels(outputs, labels)

    get_channels.sift(result_path=args.grad_path, threshold=args.theta)


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    main()
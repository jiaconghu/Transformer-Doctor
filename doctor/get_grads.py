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


class GradCalculate:
    def __init__(self, modules, num_classes):
        self.modules = [HookModule(module) for module in modules]
        self.values = [[[] for _ in range(50)] for _ in range(12)]
        self.num_classes = num_classes
        # [num_classes, num_images, channels]

    def __call__(self, outputs, labels):
        nll_loss = torch.nn.NLLLoss()(outputs, labels)
        for block in range(12):
            layer = int((11 - block) * 5 + 2)
            module = self.modules[layer]

            values = module.grads(-nll_loss, module.outputs)
            values = torch.relu(values)

            values = values.detach().cpu().numpy()

            for b in range(len(labels)):
                self.values[block][labels[b]].append(values[b])

    def sift(self, result_path, threshold):
        # layer = self.layer
        # for label in range(self.num_classes):
        #     values = self.values[label]     # (num_images, tokens, channels)
        #     values = np.asarray(values)
        #     # values = np.sum(values, axis=1)     # (num_images, channels)
        #     if len(values.shape) > 2:
        #         values = np.squeeze(values[:, 0, :])  # [num_classes, num_images, channels]

        #     values = _normalization(values, axis=1)

        #     mask = np.zeros(values.shape)
        #     mask[np.where(values > threshold)] = 1

        #     result_path = os.path.join(result_path, block)        
        #     mask_path = os.path.join(result_path, 'block_{}_layer_{}_class_{}.npy'.format(block, layer, label))
        #     np.save(mask_path, mask)
        #     print(mask_path)

        for block in range(12):
            layer = int((11 - block) * 5 + 2)
            for label in range(10):
                values = self.values[block][label]     # (num_images, tokens, channels)
                values = np.asarray(values)
                # values = np.sum(values, axis=1)     # (num_images, channels)
                if len(values.shape) > 2:
                    values = np.squeeze(values[:, 0, :])  # [num_classes, num_images, channels]

                values = _normalization(values, axis=1)

                mask = np.zeros(values.shape)
                mask[np.where(values > threshold)] = 1

                result_path_block = os.path.join(result_path, str(block))
                mask_path = os.path.join(result_path_block, 'block_{}_layer_{}_class_{}.npy'.format(block, layer, label))
                np.save(mask_path, mask)
                print(mask_path)
        


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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    data_loader = loaders.load_data(data_dir=args.data_path, data_name=args.data_name, data_type='test')

    modules = models.load_modules(model=model)

    grad_calculate = GradCalculate(modules=modules, num_classes=args.num_classes)

    # ----------------------------------------
    # forward
    # ----------------------------------------
    for i, samples in enumerate(tqdm(data_loader)):
        inputs, labels, _ = samples
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)

        grad_calculate(outputs, labels)

    grad_calculate.sift(result_path=args.grad_path, threshold=args.theta)


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    main()
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


class GradCalculate:
    def __init__(self, modules, num_classes):
        self.modules = [HookModule(module) for module in modules]
        # [num_modules, num_labels, num_images, channels]
        self.grads = [[[] for _ in range(num_classes)] for _ in range(len(modules))]
        self.activations = [[[] for _ in range(num_classes)] for _ in range(len(modules))]

    def __call__(self, outputs, labels):
        nll_loss = torch.nn.NLLLoss()(outputs, labels)
        for layer, module in enumerate(self.modules):
            if module.outputs == None:
                continue
            grads = module.grads(-nll_loss, module.outputs)
            grads = torch.relu(grads)
            grads = grads.detach().cpu().numpy()
            activations = module.outputs
            activations = torch.relu(activations)
            activations = activations.detach().cpu().numpy()
            for b in range(len(labels)):
                self.grads[layer][labels[b]].append(grads[b])
                self.activations[layer][labels[b]].append(activations[b])
    
    def sift_ac_grad(self, result_path, threshold):
        for layer, _ in enumerate(tqdm(self.activations)):
            if (layer == 2 or layer == 3):
                print('----', layer, '----activation * grad----')
                grads = self.grads[layer]
                activations = self.activations[layer]
                grads = np.asarray(grads)
                activations = np.asarray(activations)
                if len(grads.shape) == 4:
                    grads = np.squeeze(grads[:, :, 0, :])                   # [num_labels, num_images, channels]
                if len(activations.shape) == 4:
                    activations = np.squeeze(activations[:, :, 0, :])       # [num_labels, num_images, channels]

                values = grads * activations

                masks_array = []
                for value in values:
                    value = _normalization(value, axis=1)       # [num_images, channels]
                    mask = np.zeros(value.shape)                # [num_images, channels]
                    mask[np.where(value > threshold)] = 1       # [num_images, channels]
                    mask = np.sum(mask, axis=0)                 # [channels]
                    mask = np.where(mask > 3, 1, 0)             # [channels]
                    print(np.sum(mask))
                    masks_array.append(mask)
                masks = np.stack(masks_array, axis=0)
                print('masks: ', masks.shape)
                masks_path = os.path.join(result_path, 'att_ac_grad_layer_{}.npy'.format(layer))
                np.save(masks_path, masks)
    
    def sift_ac(self, result_path, threshold):
        for layer, activations in enumerate(tqdm(self.activations)):
            if (layer == 2 or layer == 3):
                print('----', layer, '----activation----')
                activations = np.asarray(activations)
                print('activations_shape: ', np.shape(activations))
                if len(activations.shape) == 4:
                    activations = np.sum(activations, axis=2)  # [num_classes, num_images, channels]

                masks_array = []
                for activation in activations:
                    activation = _normalization(activation, axis=1)       # [num_images, channels]
                    mask = np.zeros(activation.shape)                # [num_images, channels]
                    mask[np.where(activation > threshold)] = 1       # [num_images, channels]
                    mask = np.sum(mask, axis=0)                 # [channels]
                    mask = np.where(mask > 3, 1, 0)             # [channels]
                    print(np.sum(mask))
                    masks_array.append(mask)
                masks = np.stack(masks_array, axis=0)
                print('masks: ', masks.shape)
                masks_path = os.path.join(result_path, 'ac_layer_{}.npy'.format(layer))
                np.save(masks_path, masks)

    def sift_grad(self, result_path, threshold):
        for layer, grads in enumerate(tqdm(self.grads)):
            # if (layer == 2 or layer == 3):
            print('----', layer, '----grad----')
            grads = np.asarray(grads)
            if grads.shape[1] == 0:
                continue
            print('grads_shape: ', np.shape(grads))
            if len(grads.shape) == 4:
                grads = np.sum(grads, axis=2)  # [num_classes, num_images, channels]

            # grads = np.sum(grads, axis=1)  # [num_classes, channels]
            # grads = _normalization(grads, axis=1)
            # masks = np.zeros(grads.shape)
            # masks[np.where(grads > threshold)] = 1

            masks_array = []
            for grad in grads:
                grad = _normalization(grad, axis=1)       # [num_images, channels]
                mask = np.zeros(grad.shape)                # [num_images, channels]
                mask[np.where(grad > threshold)] = 1       # [num_images, channels]
                mask = np.sum(mask, axis=0)                 # [channels]
                mask = np.where(mask > 3, 1, 0)             # [channels]
                print(np.sum(mask))
                masks_array.append(mask)
            masks = np.stack(masks_array, axis=0)
            print('masks: ', masks.shape)
            masks_path = os.path.join(result_path, 'layer_{}.npy'.format(layer))
            np.save(masks_path, masks)


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', default='', type=str, help='model name')
    parser.add_argument('--data_name', default='', type=str, help='data name')
    parser.add_argument('--in_channels', default='', type=int, help='in channels')
    parser.add_argument('--num_classes', default='', type=int, help='num classes')
    parser.add_argument('--batch_size', default=512, type=int, help='batch size')
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
    # print(model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu'))['model'], strict=False))
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    model.eval()
    print(model)
    data_loader = loaders.load_data(data_dir=args.data_path, data_name=args.data_name, data_type='train', batch_size=args.batch_size)

    modules = models.load_modules(model=model)

    grad_calculate = GradCalculate(modules=modules, num_classes=args.num_classes)

    # ----------------------------------------
    # forward
    # ----------------------------------------
    for i, samples in enumerate(tqdm(data_loader)):
        inputs, labels, names = samples
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)

        grad_calculate(outputs, labels)

    # grad_calculate.sift_ac_grad(result_path=args.grad_path, threshold=args.theta)
    # grad_calculate.sift_ac(result_path=args.grad_path, threshold=args.theta)
    grad_calculate.sift_grad(result_path=args.grad_path, threshold=args.theta)


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    main()

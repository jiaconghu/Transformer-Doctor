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

    
class AttentionGet:
    def __init__(self, modules):
        self.modules = [HookModule(module) for module in modules]
        
    def __call__(self, names, outputs, labels):
        nll_loss = torch.nn.NLLLoss()(outputs, labels)
        attentions = self.modules[6].outputs        #(100, 16, 65, 65)
        
        # wall = 0.015
        # heads_num = 3
        # for i, attention in enumerate(attentions):
        #     attention_value = attention[:, 0, :]            #(16, 65, 65) -> (16, 65)
        #     attention_value = attention_value[:, 1:65]      #(16, 65) -> (16, 64)
        #     # attention_value = torch.where(attention_value < wall, 0, attention_value)
        #     attention_value = torch.sum(attention_value, dim=-1)  #(16, 64) -> (16)
        #     sorted_value, indices = torch.sort(attention_value)
        #     print(indices[0], ' ', indices[1], ' ', indices[2])
        #     tmp_attention_heads = []
        #     for j in range(heads_num):
        #         tmp = attention[indices[j]]
        #         # tmp = torch.where(tmp < wall, 0, tmp)
        #         tmp_attention_heads.append(tmp)
        #     attention_max = torch.stack(tmp_attention_heads, dim=0)
        #     attention_max = attention_max[:, 0, :]
        #     attention_max = torch.mean(attention_max, dim=0)
        #     attention_max = attention_max[1:65]
        #     attention_max = attention_max.reshape((8, 8))
        #     attention_max = attention_max.cpu().detach().numpy()
        #     save_path = os.path.join('/nfs1/ch/project/td/output/tmp/high/npys/min_npys', 'img_{}'.format(i))
        #     np.save(save_path, attention_max)

        grads = torch.autograd.grad(outputs=-nll_loss, inputs=attentions, retain_graph=True, create_graph=True)[0]
        # grads (100, 16, 65, 65)
        for i, attention in enumerate(attentions):
            # if i == 20:
            #     break
            grad = grads[i]
            grad = torch.relu(grad)             #(16, 65, 65)
            grad = grad[:, 0, :]                #(16, 65, 65) -> (16, 65)
            grad = grad[:, 1:197]                #(16, 64)
            
            # grad_sum = torch.sum(grad, dim=-1)  #(16, 64) -> (16)
            # sorted_grad, indices = torch.sort(grad_sum, descending=True)    #降序，大的在前
                  #(16, 65, 65)
            attention = attention[:, 0, :]              #(16, 65)
            attention = attention[:, 1:197]              #(16, 64)
            attention = attention * grad                #(16, 64)
            attention = torch.mean(attention, dim=0)
            attention = attention.reshape((14, 14))       #(8, 8)
            attention = attention.cpu().detach().numpy()
            save_path = os.path.join('/nfs/ch/project/td/output/tmp/low/all/low_npys', 'img_{}'.format(i))
            np.save(save_path, attention)
            print(save_path)
            # for j, head_attention in enumerate(attention):
            #     save_path = os.path.join('/nfs/ch/project/td/output/tmp/high/npys/img_{}'.format(i), 'rank_{}_head_{}.npy'.format(indices[j], j))
            #     print(save_path)
            #     head_attention = head_attention[1:65]
            #     head_attention = head_attention.reshape((8, 8))
            #     head_attention = head_attention.cpu().detach().numpy()
            #     np.save(save_path, head_attention)


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

    attention_get = AttentionGet(modules)

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
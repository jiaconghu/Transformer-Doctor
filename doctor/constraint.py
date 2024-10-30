import numpy as np
import torch
from torch import nn
from torchvision import transforms
from doctor.grad_calculate import HookModule
import os

def partial_linear(linear: nn.Linear, inp: torch.Tensor):
    weight = linear.weight.to(inp.device)  # (o, i)
    # bias = linear.bias.to(inp.device)  # (o)

    out = torch.einsum('bi,oi->boi', inp, weight)  # (b, o, i)

    # out = torch.sum(out, dim=-1)
    # out = out + bias

    return out

class Constraint:

    def __init__(self, model_name, modules, grad_path, lrp_path, alpha, beta, gamma):
        # [channel loss, choose linear layers]
        if (model_name == 'beit' or model_name == 'pvt' or model_name == 'eva'):
            self.grad_module = HookModule(modules[2])
        else:
            self.grad_module = HookModule(modules[3])
        # [attention loss, choose MyIdentity to get attention map]
        if (model_name == 'cait'):
            self.attention_module = HookModule(modules[7])
        elif (model_name == 'eva' or model_name == 'beit'):
            self.attention_module = HookModule(modules[5])
        else:
            self.attention_module = HookModule(modules[6])
        # [lrp loss]
        self.lrp_mlp_module = HookModule(modules[0])
        self.lrp_linear_module = HookModule(modules[1])
        # [channel mask, obtained from high-credit images]
        self.channels_masks = None
        if os.path.exists(grad_path):
            self.channels_masks = torch.from_numpy(np.load(grad_path)).cuda()
            print("CHANNEL MASK LOAD SUCCEED")
        else:
            print("CHANNEL MASK LOAD FAIL !!!")
        # self.lrps = torch.from_numpy(np.load(lrp_path)).cuda()
        self.grad_ratio = alpha
        self.attention_ratio = beta
        self.lrp_ratio = gamma


    def loss_grad(self, outputs, labels):
        # [low response channel loss]
        nll_loss = torch.nn.NLLLoss()(outputs, labels)
        grads = self.grad_module.grads(outputs=-nll_loss)
        grads = torch.relu(grads)
        activations = self.grad_module.outputs
        activations = torch.relu(activations)
        # channel_grads = torch.sum(grads, dim=1)
        channel_grads = grads[:, 0, :]                          # grad
        channel_ac = activations[:, 0, :]                       # activation
        channel_ac_grads = (grads * activations)[:, 0, :]       # activation*grad
        # [choose one to calculate channel loss: channel_grads, channel_ac, channel_ac_grads]
        loss = torch.sum(torch.sum(channel_grads * torch.index_select(self.channels_masks, 0, labels), dim=-1), dim=-1)
        loss = loss / len(labels)
        return loss * self.grad_ratio


    def loss_attention(self, inputs, outputs, labels, masks, patch_size, device):
        masks = masks.squeeze(1)
        masks = torch.where(masks > 0, 1, 0)
        count = 0
        for mask in masks:
            if (torch.all(mask == 1) == False):
                count = count + 1

        # [Through MyIdentity Layer to get attention data]
        sample_attentions = self.attention_module.outputs

        image_size = inputs.shape[2]
        xy_num = image_size // patch_size
        tokens = xy_num * xy_num

        # attention_actual = sample_attentions[:, :, 0, :]        # (b, h, t+1, t+1) -> (b, h, t+1)
        # attention_actual = attention_actual[:, :, 1:]           # (b, h, t+1) -> (b, h, t)

        # ====================
        # [attention mean]
        # ====================
        # attention_mean = torch.mean(attention_actual, dim=1)
        # ====================
        # [attentions * grads]
        # ====================
        nll_loss = torch.nn.NLLLoss()(outputs, labels)
        grads = torch.autograd.grad(outputs=-nll_loss, inputs=sample_attentions, retain_graph=True, create_graph=True)[0]   #(b, h, t+1, t+1)
        grads = torch.relu(grads)
        # grads = grads[:, :, 0, :]           # (b, h, t+1, t+1) -> (b, h, t+1)
        # grads = grads[:, :, 1:]             # (b, h, t+1) -> (b, h, t)
        attention_weight = sample_attentions * grads
        attention_weight = torch.sum(attention_weight, dim=2)
        attention_weight = torch.mean(attention_weight, dim=1)
        # ====================
        # [attention max]
        # ====================
        # attention_actual = torch.where(attention_actual < 0.015, 0, attention_actual)
        # attention_value = torch.sum(attention_actual, dim=-1)   # (b, h, t) -> (b, h)
        # sorted_value, indices = torch.sort(attention_value, dim=-1, descending=True)
        # indices = indices[:, 0:3]
        # attention_max_0 = attention_actual[torch.arange(attention_actual.size(0)), indices[:, 0], :]    # (b, h, t) -> (b, t)
        # attention_max_1 = attention_actual[torch.arange(attention_actual.size(0)), indices[:, 1], :]    # (b, h, t) -> (b, t)
        # attention_max_2 = attention_actual[torch.arange(attention_actual.size(0)), indices[:, 2], :]    # (b, h, t) -> (b, t)
        # attention_max = torch.stack([attention_max_0, attention_max_1, attention_max_2], dim=0)         # (b, t) -> (n, b, t)
        # attention_max = torch.mean(attention_max, dim=0)    # (n, b, t) -> (b, t)

        # masks
        # (b, t, p, p)
        sub_masks = masks.reshape(inputs.shape[0], xy_num, patch_size, xy_num, patch_size).swapaxes(2, 3).reshape(inputs.shape[0], tokens, patch_size, patch_size)
        # (b, t)
        masks_inside = torch.sum(torch.sum(sub_masks, dim=-1), dim=-1)
        # (b, t)
        attention_ideal = torch.where(masks_inside > 0, 0, 1).to(device)
        if (torch.all(attention_ideal == 1)):
            attention_ideal = torch.zeros(attention_ideal.shape)

        # real_imgs = []
        # for i, attention_save_tmp in enumerate(attention_ideal):
        #     if (torch.all(attention_save_tmp == 0) == False):
        #         real_imgs.append(i)
        #         attention_save = attention_save_tmp.reshape((7, 7))
        #         attention_save = attention_save.cpu().detach().numpy()
        #         folder_path = '/nfs1/ch/project/td/output/tmp/high/masks/npys'
        #         my_count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
        #         save_path = folder_path + '/' + str(my_count) + '.npy'
        #         np.save(save_path, attention_save)
        # print(real_imgs)
        # print('DONE')

        loss = torch.sum(torch.sum(attention_ideal * attention_weight, dim=-1), dim=-1)
        if (count != 0):
            loss = loss / count
        return loss * self.attention_ratio

    def loss_lrp(self, labels):
        mlp = self.lrp_mlp_module.module
        mlp_inputs = self.lrp_mlp_module.inputs
        mlp_head = partial_linear(mlp, mlp_inputs)      # mlp_head: (b, o, i)
        mlp_head = torch.relu(mlp_head)
        linear2 = self.lrp_linear_module.module
        linear_inputs = self.lrp_linear_module.inputs
        linear_layer = partial_linear(linear2, linear_inputs)      # linear_layer: (128, 1024, 2048)
        batch_size = len(labels)

        # only correct label to calculate mlp_inputs lrp
        mlp_lrp_inputs = mlp_head[torch.arange(mlp_head.size(0)), labels, :]        # (b, o, i) ---(labels(b))--> (b, i)

        # lrps: (labels_num, i)   max-min normalization
        max_lrp = torch.max(self.lrps, dim=-1).values
        min_lrp = torch.min(self.lrps, dim=-1).values
        tmp = max_lrp - min_lrp
        my_lrp = (self.lrps - min_lrp.unsqueeze(-1)) / tmp.unsqueeze(-1)     # my_lrp (labels_num, i)

        ideal_lrps = my_lrp[labels]     # (labels_num, i) -> (b, i)
        # sift the lrp data < 0.2 to 0
        ideal_lrps = torch.where(ideal_lrps < 0.20, 0, ideal_lrps)
        # reverse 1 and 0
        ideal_lrps = torch.where(ideal_lrps == 0, 1, 0)
        loss_tmp = torch.sum(mlp_lrp_inputs * ideal_lrps, dim=-1)
        loss_tmp = loss_tmp / batch_size
        loss = torch.sum(loss_tmp, dim=-1)
        return loss * self.lrp_ratio


def calculate_loss_channel(channels, grads, labels):
    grads = torch.relu(grads)
    channel_grads = torch.sum(grads, dim=1)  # [batch_size, channels]
    loss = torch.sum(torch.sum(channel_grads * torch.index_select(channels, 0, labels), dim=-1), dim=-1)
    loss = loss / len(labels)
    return loss

        # second linear
        # linear_sum = torch.sum(linear_layer, dim=-1)                                            #(128, 1024)
        # linear_sum = linear_sum.unsqueeze(2)                                                    #(128, 1024,)
        # linear_sum = linear_sum.repeat(1, 1, linear_layer.shape[2])                             #(128, 1024, 2048)
        # linear_ratio = linear_layer / linear_sum                                                #(128, 1024, 2048)
        # mlp_inputs_repeat = mlp_lrp_inputs.unsqueeze(2).repeat(1, 1, linear_layer.shape[2])     #(128, 1024, 2048)
        # linear_lrp_inputs = torch.sum(linear_ratio * mlp_inputs_repeat, dim=1)                  #(128, 2048)
        # second linear

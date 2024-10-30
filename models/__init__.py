import torch
from models import vit
from models.cait import cait_xxs24_imagenet, cait_xxs24_cifar
from models.deit import deit_tiny_imagenet, deit_tiny_cifar
from models.pvt import pvt_tiny_imagenet, pvt_tiny_cifar
from models.tnt import tnt_s_imagenet, tnt_s_cifar
from models.tvit import vit_tiny_imagenet, vit_tiny_cifar
from models.beit import beit_base_patch16_224, beit_base_patch16_cifar
from models.eva import eva02_tiny_patch14_224, eva02_tiny_patch14_cifar

import random
import numpy as np
# import timm
from models.tvit import MyIdentity


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_model(model_name, data_name, in_channels=3, num_classes=10):
    print('-' * 50)
    print('MODEL NAME:', model_name)
    print('NUM CLASSES:', num_classes)
    print('-' * 50)

    model_kwargs = dict(drop_path_rate=0.1)
    model_kwargs['num_classes'] = num_classes
    print(model_kwargs)

    model = None
    if model_name == 'tvit' and ('cifar' in data_name):
        model = vit_tiny_cifar(**model_kwargs)
    elif model_name == 'tvit' and ('imagenet' in data_name):
        model = vit_tiny_imagenet(**model_kwargs)
    elif model_name == 'cait' and ('cifar' in data_name):
        model = cait_xxs24_cifar(**model_kwargs)
    elif model_name == 'cait' and ('imagenet' in data_name):
        model = cait_xxs24_imagenet(**model_kwargs)
    elif model_name == 'pvt' and ('cifar' in data_name):
        model = pvt_tiny_cifar(**model_kwargs)
    elif model_name == 'pvt' and ('imagenet' in data_name):
        model = pvt_tiny_imagenet(**model_kwargs)
    elif model_name == 'deit' and ('cifar' in data_name):
        model = deit_tiny_cifar(**model_kwargs)
    elif model_name == 'deit' and ('imagenet' in data_name):
        model = deit_tiny_imagenet(**model_kwargs)
    elif model_name == 'tnt' and ('cifar' in data_name):
        model = tnt_s_cifar(**model_kwargs)
    elif model_name == 'tnt' and ('imagenet' in data_name):
        model = tnt_s_imagenet(**model_kwargs)
    elif model_name == 'beit' and ('imagenet' in data_name):
        model = beit_base_patch16_224(**model_kwargs)
    elif model_name == 'beit' and ('cifar' in data_name):
        model = beit_base_patch16_cifar(**model_kwargs)
    elif model_name == 'eva' and ('imagenet' in data_name):
        model = eva02_tiny_patch14_224(**model_kwargs)
    elif model_name == 'eva' and ('cifar' in data_name):
        model = eva02_tiny_patch14_cifar(**model_kwargs)
    
    return model


def load_modules(model, model_layers=None):
    assert model_layers is None or type(model_layers) is list

    modules = []
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            modules.append(module)
        elif isinstance(module, MyIdentity):
            modules.append(module)

    modules.reverse()  # reverse order
    if model_layers is None:
        model_modules = modules
        # model_modules = modules[:10]
    else:
        model_modules = []
        for layer in model_layers:
            model_modules.append(modules[layer])

    print('-' * 50)
    print('Model Layers:', model_layers)
    print('Model Modules:', model_modules)
    print('Model Modules Length:', len(model_modules))
    print('-' * 50)

    return model_modules


if __name__ == '__main__':
    from torchsummary import summary

    model = load_model('vgg16')
    print(model)
    # summary(model, (3, 224, 224))

    modules = load_modules(model)

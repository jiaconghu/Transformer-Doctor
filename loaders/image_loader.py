from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision import datasets
# from torchvision.datasets import ImageFolder
from loaders.image_dataset import ImageDataset
from loaders.image_dataset import ImageMaskDataset
from loaders.image_masks_transforms import Normalize, ToTensor, Resize, RandomHorizontalFlip, Compose
from loaders.data_enhance import create_transform, create_transform_mask

def _build_timm_aug_kwargs(image_size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    train_aug_kwargs = dict(input_size=image_size, is_training=True, use_prefetcher=False, no_aug=False,
                            scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), hflip=0.5, vflip=0., color_jitter=0.4,
                            auto_augment='rand-m9-mstd0.5-inc1', interpolation='random', mean=mean, std=std,
                            re_prob=0.25, re_mode='pixel', re_count=1, re_num_splits=0, separate=False)

    eval_aug_kwargs = dict(input_size=image_size, is_training=False, use_prefetcher=False, no_aug=False, crop_pct=0.875,
                           interpolation='bilinear', mean=mean, std=std)

    return {
        'train_aug_kwargs': train_aug_kwargs,
        'eval_aug_kwargs': eval_aug_kwargs
    }

cifar10_train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize((0.4913, 0.4821, 0.4465),
    #                      (0.2470, 0.2434, 0.2615)),
    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                         (0.2023, 0.1994, 0.2010)),
])

cifar10_test_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    # transforms.Normalize((0.4913, 0.4821, 0.4465),
    #                      (0.2470, 0.2434, 0.2615)),
    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                         (0.2023, 0.1994, 0.2010)),
])

cifar100_train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4913, 0.4821, 0.4465),
                         (0.2470, 0.2434, 0.2615)),
])

cifar100_test_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4913, 0.4821, 0.4465),
                         (0.2470, 0.2434, 0.2615)),
])

imagenet_train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

imagenet_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

imagenet_mask_train_transform = Compose([
    Resize((224, 224)),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

imagenet_mask_test_transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
aug_kwargs = _build_timm_aug_kwargs(224, mean, std)
imagenet_train_transform_enhance = create_transform(**aug_kwargs['train_aug_kwargs'])
imagenet_test_transform_enhance = create_transform(**aug_kwargs['eval_aug_kwargs'])
imagenet_mask_train_transform_enhance = create_transform_mask(**aug_kwargs['train_aug_kwargs'])
imagenet_mask_test_transform_enhance = create_transform(**aug_kwargs['eval_aug_kwargs'])


def _get_set(data_path, transform):
    return ImageDataset(image_dir=data_path,
                        transform=transform)

def _get_set_mask(data_path, transform, mask_path):
    return ImageMaskDataset(image_dir=data_path,
                            mask_dir=mask_path,
                            transform=transform)


def load_images(data_dir, data_name, data_type=None, batch_size=512):
    assert data_name in ['cifar10', 'cifar100', 'imagenet', 'imagenet50', 'imagenet10', 'imagenet300', 'imagenet100']
    assert data_type is None or data_type in ['train', 'test']

    data_transform = None
    if data_name == 'cifar10' and data_type == 'train':
        data_transform = cifar10_train_transform
    elif data_name == 'cifar10' and data_type == 'test':
        data_transform = cifar10_test_transform
    elif data_name == 'cifar100' and data_type == 'train':
        data_transform = cifar100_train_transform
    elif data_name == 'cifar100' and data_type == 'test':
        data_transform = cifar100_test_transform
    elif ('imagenet' in data_name) and data_type == 'train':
        data_transform = imagenet_train_transform
    elif ('imagenet' in data_name) and data_type == 'test':
        data_transform = imagenet_test_transform
    assert data_transform is not None

    print(data_transform)

    data_set = _get_set(data_dir, transform=data_transform)
    # data_set = datasets.ImageFolder(root=data_dir,
    #                             transform=imagenet_test_transform)
    data_loader = DataLoader(dataset=data_set,
                             batch_size=batch_size,
                             num_workers=8,
                             shuffle=True)

    return data_loader


def load_images_masks(data_dir, data_name, mask_dir, data_type=None, batch_size=512):
    assert data_name in ['cifar10', 'cifar100', 'imagenet', 'imagenet50', 'imagenet10', 'imagenet300', 'imagenet100']
    assert data_type is None or data_type in ['train', 'test']

    data_transform = None
    if data_name == 'cifar10' and data_type == 'train':
        data_transform = cifar10_train_transform
    elif data_name == 'cifar10' and data_type == 'test':
        data_transform = cifar10_test_transform
    elif ('imagenet' in data_name) and data_type == 'train':
        data_transform = imagenet_mask_train_transform
    elif ('imagenet' in data_name) and data_type == 'test':
        data_transform = imagenet_mask_test_transform
    assert data_transform is not None

    data_set = _get_set_mask(data_dir, transform=data_transform, mask_path=mask_dir)
    # data_set = datasets.ImageFolder(root=data_dir,
    #                             transform=imagenet_test_transform)
    data_loader = DataLoader(dataset=data_set,
                             batch_size=batch_size,
                             num_workers=8,
                             shuffle=True)

    return data_loader

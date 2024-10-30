from loaders.image_loader import load_images
from loaders.image_loader import load_images_masks


def load_data(data_dir, data_name, data_type, batch_size):
    print('-' * 50)
    print('DATA PATH:', data_dir)
    print('DATA NAME:', data_name)
    print('DATA TYPE:', data_type)
    print('-' * 50)

    return load_images(data_dir, data_name, data_type, batch_size)
    

def load_data_mask(data_dir, data_name, mask_dir, data_type, batch_size):
    print('-' * 50)
    print('DATA PATH:', data_dir)
    print('DATA NAME:', data_name)
    print('MASK PATH:', mask_dir)
    print('DATA TYPE:', data_type)

    print('-' * 50)

    return load_images_masks(data_dir, data_name, mask_dir, data_type, batch_size)
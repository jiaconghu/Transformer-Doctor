import os
import json
import numpy as np
import cv2


def save_cv(img, path):
    print(path)
    img_dir, _ = os.path.split(path)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    cv2.imwrite(path, img)


def to_mask(img_shape):
    """
    边界点生成mask
    :param img_shape: [h,w]
    :param polygons: labelme JSON中的边界点格式 [[x1,y1],[x2,y2],[x3,y3],...[xn,yn]]
    :return: mask 0-1
    """
    mask = np.ones(img_shape, dtype=np.uint8)
    return mask


def main():
    mask = to_mask([256, 256]) * 255
    mask_path = os.path.join('/nfs/ch/project/td/output/ideal/default', 'default.png')
    save_cv(mask, mask_path)
    # images_dir = '/nfs3-p1/hjc/datasets/imagenet/ch_select_test/'
    # for root, _, files in os.walk(images_dir):
    #     for file in files:
    #         if os.path.splitext(file)[1] == '.JPEG':
    #             img_path = os.path.join(root, file)
    #             img = cv2.imread(img_path)

    #             json_name = os.path.splitext(file)[0]

    #             mask = to_mask([img.shape[0], img.shape[1]]) * 255
    #             # print(mask.shape)
    #             mask_path = os.path.join('/nfs/ch/project/td/output/ideal/default', 'default.png')
    #             save_cv(mask, mask_path)


if __name__ == '__main__':
    main()
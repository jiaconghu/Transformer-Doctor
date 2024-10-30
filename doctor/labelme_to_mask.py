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


def parse_json(json_path):
    data = json.load(open(json_path))
    shapes = data['shapes'][0]
    points = shapes['points']
    return points


def polygons_to_mask(img_shape, polygons):
    mask = np.zeros(img_shape, dtype=np.uint8)
    polygons = np.asarray([polygons], np.int32)
    cv2.fillPoly(mask, polygons, 1)
    return mask


def main():
    images_dir = 'xxx'
    for root, _, files in os.walk(images_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.json':
                img_path = os.path.join(root, file.replace('json', 'JPEG'))
                img = cv2.imread(img_path)

                json_name = os.path.splitext(file)[0]
                json_path = os.path.join(root, file)

                mask = polygons_to_mask([img.shape[0], img.shape[1]], parse_json(json_path)) * 255
                print(json_name)
                print(mask.shape)
                mask_path = os.path.join('xxx', json_name + '.png')
                print(mask)
                save_cv(mask, mask_path)


if __name__ == '__main__':
    main()
import os
import torch
import argparse
from tqdm import tqdm
# import timm

import loaders
import models
from utils import file_util


class SampleSift:
    def __init__(self, num_classes, num_samples, is_high_confidence=True):
        self.names = [[None for j in range(num_samples)] for i in range(num_classes)]
        self.scores = torch.zeros((num_classes, num_samples))
        self.nums = torch.zeros(num_classes, dtype=torch.long)
        self.is_high_confidence = is_high_confidence

    def __call__(self, outputs, labels, names):
        softmax = torch.nn.Softmax(dim=1)(outputs.detach())
        scores, predicts = torch.max(softmax, dim=1)
        # print(scores)

        for i, label in enumerate(labels):
            if self.is_high_confidence == 1:
                print('-->high')
                if self.nums[label] == self.scores.shape[1]:
                    score_min, index = torch.min(self.scores[label], dim=0)
                    if scores[i] > score_min:
                        self.scores[label][index] = scores[i]
                        self.names[label.item()][index.item()] = names[i]
                else:
                    self.scores[label][self.nums[label]] = scores[i]
                    self.names[label.item()][self.nums[label].item()] = names[i]
                    self.nums[label] += 1
            else:  # sift low confidence
                print('-->low')
                if self.nums[label] == self.scores.shape[1]:
                    score_max, index = torch.max(self.scores[label], dim=0)
                    if scores[i] < score_max:
                        self.scores[label][index] = scores[i]
                        self.names[label.item()][index.item()] = names[i]
                else:
                    self.scores[label][self.nums[label]] = scores[i]
                    self.names[label.item()][self.nums[label].item()] = names[i]
                    self.nums[label] += 1

    def save_image(self, input_path, output_path):

        class_names = sorted([d.name for d in os.scandir(input_path) if d.is_dir()])
        print(self.names)

        for label, image_list in enumerate(self.names):
            for image in tqdm(image_list):
                class_name = class_names[label]

                src_path = os.path.join(input_path, class_name, str(image))
                dst_path = os.path.join(output_path, class_name, str(image))
                file_util.copy_file(src_path, dst_path)


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', default='', type=str, help='model name')
    parser.add_argument('--data_name', default='', type=str, help='data name')
    parser.add_argument('--num_classes', default='', type=int, help='num classes')
    parser.add_argument('--batch_size', default=512, type=int, help='batch size')
    parser.add_argument('--model_path', default='', type=str, help='model path')
    parser.add_argument('--data_dir', default='', type=str, help='data dir')
    parser.add_argument('--save_dir', default='', type=str, help='sift dir')
    parser.add_argument('--num_samples', default=10, type=int, help='num samples')
    parser.add_argument('--is_high_confidence', default=1, type=int, help='is high confidence')
    args = parser.parse_args()

    # ----------------------------------------
    # basic configuration
    # ----------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    print('-' * 50)
    print('TRAIN ON:', device)
    print('MODEL PATH:', args.model_path)
    print('DATA PATH:', args.data_dir)
    print('RESULT PATH:', args.save_dir)
    print('-' * 50)

    # ----------------------------------------
    # model/data configuration
    # ----------------------------------------
    model = models.load_model(model_name=args.model_name, data_name=args.data_name, num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    model.eval()

    data_loader = loaders.load_data(data_dir=args.data_dir, data_name=args.data_name, data_type='train',
                                    batch_size=args.batch_size)
    print('DATA LOAD DONE')

    sample_sift = SampleSift(num_classes=args.num_classes, num_samples=args.num_samples,
                             is_high_confidence=args.is_high_confidence)

    print('SAMPLE SIFT DONE')

    # ----------------------------------------
    # forward
    # ----------------------------------------
    for samples in tqdm(data_loader):
        inputs, labels, names = samples
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            sample_sift(outputs=outputs, labels=labels, names=names)

    sample_sift.save_image(args.data_dir, args.save_dir)


if __name__ == '__main__':
    main()

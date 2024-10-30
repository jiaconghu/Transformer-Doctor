#!/bin/bash
export PYTHONPATH=[Your custom-defined path]
export CUDA_VISIBLE_DEVICES=1
export result_path='[Your custom-defined path]'
#export exp_name='vgg16d_cifar10_mu'
#export model_name='vgg16d'
export exp_name='vit_high_test'
export model_name='vit_linear'
export data_name='cifar10'
export num_classes=10
export model_path='[Your custom-defined path]'
export data_dir='[Your custom-defined path]'
export save_dir=${result_path}'/'${exp_name}
python doctor/get_linear.py \
  --model_name ${model_name} \
  --data_name ${data_name} \
  --num_classes ${num_classes} \
  --model_path ${model_path} \
  --data_dir ${data_dir} \
  --save_dir ${save_dir}

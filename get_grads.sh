#!/bin/bash
export result_path='[Your custom-defined path]'
export exp_name='ideal'
export model_name='tvit'
export data_name='imagenet50'
export in_channels=3
export num_classes=50
export model_path='[Your custom-defined path]'
export data_dir='[Your custom-defined path]'
# export grad_path=${result_path}${exp_name}'/visualize/'${data_name}'/grad'
export grad_path='[Your custom-defined path]'
export theta=0.4
export device_index='0'
python doctor/get_grads.py \
  --model_name ${model_name} \
  --data_name ${data_name} \
  --in_channels ${in_channels} \
  --num_classes ${num_classes} \
  --model_path ${model_path} \
  --data_path ${data_dir} \
  --grad_path ${grad_path} \
  --theta ${theta} \
  --device_index ${device_index}

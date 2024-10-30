#!/bin/bash
export result_path='[Your custom-defined path]'
export device_index='2'
export exp_name='ideal'
export model_name='tvit'
export data_name='cifar10'
export in_channels=3
export num_classes=10
export batch_size=512
export model_path=${result_path}'/train_models/pth/'${data_name}'/'${model_name}'_'${data_name}'_base.pth'
# export model_path=${result_path}'/baseline/'${model_name}'/checkpoint.pth'
export data_dir=${result_path}'/'${exp_name}'/high_images/'${model_name}'/'${data_name}
export grad_path=${result_path}${exp_name}'/grads/'${model_name}'/'${data_name}
export theta=0.15
python doctor/grad_calculate.py \
  --model_name ${model_name} \
  --data_name ${data_name} \
  --in_channels ${in_channels} \
  --num_classes ${num_classes} \
  --model_path ${model_path} \
  --data_path ${data_dir} \
  --grad_path ${grad_path} \
  --theta ${theta} \
  --batch_size ${batch_size} \
  --device_index ${device_index}

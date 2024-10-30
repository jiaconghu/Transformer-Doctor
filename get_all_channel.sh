#!/bin/bash
export PYTHONPATH=[Your custom-defined path]
export CUDA_VISIBLE_DEVICES=1
#export result_path='[Your custom-defined path]'
#export exp_name='tmp'
export model_name='tvit'
export data_name='imagenet'
export in_channels=3
export num_classes=1000
export model_path=[Your custom-defined path]
#export data_dir=[Your custom-defined path]
#export save_dir=[Your custom-defined path]
export data_dir=[Your custom-defined path]
export save_dir=[Your custom-defined path]
#export data_dir=[Your custom-defined path]
#export save_dir=[Your custom-defined path]
export theta=0.15
export device_index='0'
python doctor/get_all_channel.py \
  --model_name ${model_name} \
  --data_name ${data_name} \
  --in_channels ${in_channels} \
  --num_classes ${num_classes} \
  --model_path ${model_path} \
  --data_path ${data_dir} \
  --grad_path ${save_dir} \
  --theta ${theta} \
  --device_index ${device_index}

#!/bin/bash
export PYTHONPATH=[Your custom-defined path]
export CUDA_VISIBLE_DEVICES=1
export result_path='[Your custom-defined path]'
export exp_name='ideal'
export model_name='tvit'
export data_name='imagenet'
export num_classes=1000
export model_path=${result_path}'/train_models/pth/'${data_name}'/tvit_'${data_name}'_all.pth'
# export data_dir=${result_path}'/'${exp_name}'/low_images/tvit/'${data_name}
export data_dir='[Your custom-defined path]'
export save_dir=${result_path}'/tmp/attentions/tvit/npys'
python doctor/get_attention.py \
  --model_name ${model_name} \
  --data_name ${data_name} \
  --num_classes ${num_classes} \
  --model_path ${model_path} \
  --data_dir ${data_dir} \
  --save_dir ${save_dir}
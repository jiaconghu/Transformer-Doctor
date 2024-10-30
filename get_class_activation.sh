#!/bin/bash
export PYTHONPATH=[Your custom-defined path]
export CUDA_VISIBLE_DEVICES=0
export result_path='[Your custom-defined path]'
export exp_name='ideal'
export model_name='tvit'
export data_name='imagenet'
export num_classes=1000
export model_path=${result_path}'/train_models/pth/'${data_name}'/'${model_name}'_'${data_name}'_base.pth'
export data_dir=${result_path}'/'${exp_name}'/low_images/'${model_name}'/'${data_name}
export save_dir=${result_path}'/tmp/class_activations/'${model_name}'/low/npys'
python doctor/get_class_activation.py \
  --model_name ${model_name} \
  --data_name ${data_name} \
  --num_classes ${num_classes} \
  --model_path ${model_path} \
  --data_dir ${data_dir} \
  --save_dir ${save_dir}

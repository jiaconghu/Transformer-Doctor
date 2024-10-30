#!/bin/bash
export PYTHONPATH=[Your custom-defined path]
export CUDA_VISIBLE_DEVICES=1
#export result_path='[Your custom-defined path]'
#export exp_name='ideal'
export model_name='tvit'
export data_name='imagenet'
export num_classes=1000
export model_path=[Your custom-defined path]
#export data_dir=[Your custom-defined path]
#export save_dir=[Your custom-defined path]
#export data_dir=[Your custom-defined path]
#export save_dir=[Your custom-defined path]
export data_dir=[Your custom-defined path]
export save_dir=[Your custom-defined path]
python doctor/get_all_attention.py \
  --model_name ${model_name} \
  --data_name ${data_name} \
  --num_classes ${num_classes} \
  --model_path ${model_path} \
  --data_dir ${data_dir} \
  --save_dir ${save_dir}

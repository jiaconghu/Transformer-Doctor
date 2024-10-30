#!/bin/bash
export PYTHONPATH=/vipa-nfs/homes/hjc/projects/TD/codes
export CUDA_VISIBLE_DEVICES=1
#export result_dir='/mnt/nfs/hjc/project/td/output'
export exp_name='ideal'
export model_name='tvit'
export data_name='imagenet10'
export num_classes=10
export batch_size=512
export model_path='/mnt/nfs/hjc/project/td/output/train_models/pth/'${data_name}'/'${model_name}'_'${data_name}'_base.pth'
export data_dir='/mnt/nfs/hjc/project/td/dataset/'${data_name}'/train'
#export save_dir='/vipa-nfs/homes/hjc/projects/TD/outputs/'${exp_name}'/high_train/'${model_name}'/'${data_name}
export save_dir='/vipa-nfs/homes/hjc/projects/TD/outputs/'${exp_name}'/low_train/'${model_name}'/'${data_name}
export num_samples=10
export is_high_confidence=0
python doctor/sample_sift.py \
  --model_name ${model_name} \
  --data_name ${data_name} \
  --num_classes ${num_classes} \
  --model_path ${model_path} \
  --data_dir ${data_dir} \
  --save_dir ${save_dir} \
  --num_samples ${num_samples} \
  --batch_size ${batch_size} \
  --is_high_confidence ${is_high_confidence}

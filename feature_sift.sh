#!/bin/bash
export PYTHONPATH=/nfs3-p1/ch/project/td/codes
export CUDA_VISIBLE_DEVICES=0
# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_USE_CUDA_DSA=1
export result_path='/nfs3-p1/ch/project/td/output'
#export exp_name='vgg16_cifar10_pure'
#export model_name='vgg16'
export exp_name='vitb16'
export model_name='vitb16py'
export data_name='imagenet'
export num_classes=1
export model_path=${result_path}'/'${exp_name}'/models/model.safetensors'
export data_dir='/nfs3-p1/ch/project/td/output/vitb16/images/chtrain_vitb16'
#export data_dir='/nfs3-p1/hjc/datasets/cifar10/test'
#export data_dir=${result_path}'/'${exp_name}'/adv_images/PGD/test'
export save_dir=${result_path}'/'${exp_name}'/features'
export num_samples=50
python doctor/feature_sift.py \
  --model_name ${model_name} \
  --data_name ${data_name} \
  --num_classes ${num_classes} \
  --model_path ${model_path} \
  --data_dir ${data_dir} \
  --save_dir ${save_dir} \
  --num_samples ${num_samples}

#echo "STRAT !!!"
#export PYTHONPATH=/nfs3-p1/hjc/projects/RO/codes
#export CUDA_VISIBLE_DEVICES=0
#export result_path='/nfs3-p1/hjc/projects/RO/outputs'
#export model_name='vgg16'
#export data_name='cifar10'
##export data_dir='/nfs3-p1/hjc/datasets/cifar10/train'
#export data_dir=${result_path}'/'${exp_name}'/adv_images/PGD/train'
#export num_classes=10
#export num_samples=5000
#exp_names=(
#  #'vgg16d_cifar10_$04041006b'
#  #'vgg16d_cifar10_$04041006c'
#  #'vgg16d_cifar10_$04041006f'
#  #'vgg16d_cifar10_$04041006g'
#  #'vgg16d_cifar10_$04042324a'
#  #'vgg16d_cifar10_$04042324b'
#  'vgg16d_cifar10_04162018c7'
#  #'vgg16d_cifar10_04162018c8'
#)
#for exp_name in ${exp_names[*]}; do
#  export model_path=${result_path}'/'${exp_name}'/models/model_ori.pth'
#  export save_dir=${result_path}'/'${exp_name}'/features'
#  python doctor/feature_sift.py \
#    --model_name ${model_name} \
#    --data_name ${data_name} \
#    --num_classes ${num_classes} \
#    --model_path ${model_path} \
#    --data_dir ${data_dir} \
#    --save_dir ${save_dir} \
#    --num_samples ${num_samples} &
#done
#wait
#echo "ACCOMPLISH ！！！"

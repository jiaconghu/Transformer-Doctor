export PYTHONPATH="[Your custom-defined path]"
export CUDA_VISIBLE_DEVICES=1
export result_dir='[Your custom-defined path]/output'
export exp_name='hjc_test'
export data_name='imagenet10'
export num_classes=10
# export data_dir='[Your custom-defined path]/output/ideal/high_images/vit/'${data_name}
export data_dir='[Your custom-defined path]/dataset/'${data_name}'/test'
#export data_dir=${result_dir}'/'${exp_name}'/low_images/'${model_name}'/'${data_name}
export save_dir=${result_dir}'/'${exp_name}
#export model_names='tvit'
export model_names='deit'
export model_paths='[Your custom-defined path]/output/train_models/pth/'${data_name}'/'${model_names}'_'${data_name}'_base.pth'

python doctor/test.py \
    --model_name ${model_names} \
    --data_name ${data_name} \
    --num_classes ${num_classes} \
    --model_path ${model_paths} \
    --data_dir ${data_dir} \
    --save_dir ${save_dir}

export PYTHONPATH="[Your custom-defined path]"
export CUDA_VISIBLE_DEVICES=1
export result_dir='[Your custom-defined path]/output'
export exp_name='hjc_test'
export data_name='imagenet10'
export num_classes=10
#export data_dir='[Your custom-defined path]/dataset/'${data_name}'/test'
export data_dir='[Your custom-defined path]/ideal/low_test/tvit/imagenet10'
export save_dir=${result_dir}'/'${exp_name}
export model_names='tvit'
#export model_names='deit'
export model_paths='[Your custom-defined path]/output/train_models/pth/'${data_name}'/'${model_names}'_'${data_name}'_base.pth'

python core/test_mask.py \
    --model_name ${model_names} \
    --data_name ${data_name} \
    --num_classes ${num_classes} \
    --model_path ${model_paths} \
    --data_dir ${data_dir} \
    --save_dir ${save_dir}

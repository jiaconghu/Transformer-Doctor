export PYTHONPATH="[Your custom-defined path]"
export CUDA_VISIBLE_DEVICES=0
export result_dir='[Your custom-defined path]/output'
export exp_name='train_models'
export data_name='imagenet10'
export num_classes=10
export num_epochs=300
export model_name='tvit'
export train_data_dir='[Your custom-defined path]/dataset/'${data_name}'/train'
export test_data_dir='[Your custom-defined path]/dataset/'${data_name}'/test'
export log_dir=${result_dir}'/'${exp_name}'/tensorboard/'${data_name}'/'${model_name}'_'${data_name}'_baseline_test'
export model_path=${result_dir}'/'${exp_name}'/pth/'${data_name}

python doctor/train_1k.py \
    --model_name ${model_name} \
    --data_name ${data_name} \
    --num_classes ${num_classes} \
    --num_epochs ${num_epochs} \
    --model_dir ${model_path} \
    --train_data_dir ${train_data_dir} \
    --test_data_dir ${test_data_dir} \
    --log_dir ${log_dir}

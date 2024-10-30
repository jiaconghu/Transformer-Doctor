export PYTHONPATH="[Your custom-defined path]"
export CUDA_VISIBLE_DEVICES=2
export result_dir='[Your custom-defined path]/output'
export exp_name='train_models'
export data_name='cifar10'
export num_classes=10
export num_epochs=300
export batch_size=512
export model_name='tvit'
export train_data_dir='[Your custom-defined path]/dataset/'${data_name}'/train'
export test_data_dir='[Your custom-defined path]/dataset/'${data_name}'/test'
export result_name=${model_names}'_'${data_name}'_base'
export log_dir=${result_dir}'/'${exp_name}'/tensorboard/'${data_name}'/'${result_name}
export model_path=${result_dir}'/'${exp_name}'/pth/'${data_name}

python doctor/train.py \
    --model_name ${model_name} \
    --data_name ${data_name} \
    --num_classes ${num_classes} \
    --num_epochs ${num_epochs} \
    --batch_size ${batch_size} \
    --model_dir ${model_path} \
    --train_data_dir ${train_data_dir} \
    --test_data_dir ${test_data_dir} \
    --result_name ${result_name} \
    --log_dir ${log_dir}

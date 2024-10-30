export PYTHONPATH="[Your custom-defined path]"
export CUDA_VISIBLE_DEVICES=0
export result_dir='[Your custom-defined path]/output'
export exp_name='train_models'
export data_name='imagenet50'
export num_classes=50
export num_epochs=100
export model_names='vit'
export origin_path=${result_dir}'/'${exp_name}'/pth/'${data_name}'/'${model_names}'_'${data_name}'_base.pth'
export train_data_dir='[Your custom-defined path]/dataset/'${data_name}'/train'
export test_data_dir='[Your custom-defined path]/dataset/'${data_name}'/test'
export log_dir=${result_dir}'/'${exp_name}'/tensorboard/'${data_name}'/'${model_names}'_'${data_name}'_attlrp'
export pth_path=${result_dir}'/'${exp_name}'/pth/'${data_name}
export grad_path=${result_dir}'/ideal/grads/'${model_names}'/'${data_name}'/layer_2.npy'
export lrp_path=${result_dir}'/ideal/lrps/'${model_names}'/'${data_name}'/result.npy'
export mask_path=${result_dir}'/ideal/atts/'${data_name}

python doctor/train_plus.py \
    --model_name ${model_names} \
    --data_name ${data_name} \
    --num_classes ${num_classes} \
    --num_epochs ${num_epochs} \
    --origin_dir ${origin_path}\
    --model_dir ${pth_path} \
    --train_data_dir ${train_data_dir} \
    --test_data_dir ${test_data_dir} \
    --log_dir ${log_dir}\
    --grad_dir ${grad_path}\
    --lrp_dir ${lrp_path}\
    --mask_dir ${mask_path}

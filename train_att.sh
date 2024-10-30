export PYTHONPATH="[Your custom-defined path]"
export CUDA_VISIBLE_DEVICES=3
export result_dir='[Your custom-defined path]/output'
export exp_name='train_models'
export data_name='imagenet10'
export num_classes=10
export num_epochs=100
export batch_size=512
export model_names='eva'
export train_data_dir='[Your custom-defined path]/dataset/'${data_name}'/train'
# export train_data_dir=${result_dir}'/ideal/high_images/tvit/'${data_name}
export test_data_dir='[Your custom-defined path]/dataset/'${data_name}'/test'
export origin_path=${result_dir}'/'${exp_name}'/pth/'${data_name}'/'${model_names}'_'${data_name}'_base.pth'
export result_name=${model_names}'_'${data_name}'_att'
export log_dir=${result_dir}'/'${exp_name}'/tensorboard/'${data_name}'/'${result_name}
export model_path=${result_dir}'/'${exp_name}'/pth/'${data_name}
export grad_path=${result_dir}'/ideal/grads/'${model_names}'/'${data_name}'/layer_3.npy'
export lrp_path=${result_dir}'/ideal/lrps/'${model_names}'/'${data_name}'/result.npy'
export mask_path=${result_dir}'/ideal/atts/'${data_name}

python doctor/train_att.py \
    --model_name ${model_names} \
    --data_name ${data_name} \
    --num_classes ${num_classes} \
    --num_epochs ${num_epochs} \
    --batch_size ${batch_size} \
    --result_name ${result_name} \
    --origin_dir ${origin_path}\
    --model_dir ${model_path} \
    --train_data_dir ${train_data_dir} \
    --test_data_dir ${test_data_dir} \
    --log_dir ${log_dir} \
    --grad_dir ${grad_path}\
    --lrp_dir ${lrp_path}\
    --mask_dir ${mask_path}

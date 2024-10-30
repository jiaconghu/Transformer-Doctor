export PYTHONPATH="[Your custom-defined path]"
export CUDA_VISIBLE_DEVICES=2
export result_dir='[Your custom-defined path]/output'
export exp_name='train_models'
export data_name='cifar10'
export num_classes=10
export num_epochs=100
export batch_size=512
export model_names='tvit'
export train_data_dir='[Your custom-defined path]/dataset/'${data_name}'/train'
export test_data_dir='[Your custom-defined path]/dataset/'${data_name}'/test'
export origin_path=${result_dir}'/'${exp_name}'/pth/'${data_name}'/'${model_names}'_'${data_name}'_base.pth'
# export origin_path=${result_dir}'/baseline/'${model_names}'/checkpoint.pth'
export i=0
export layer=$((i*4+3))
export result_name=${model_names}'_'${data_name}'_grad'
export log_dir=${result_dir}'/'${exp_name}'/tensorboard/'${data_name}'/'${result_name}
export model_path=${result_dir}'/'${exp_name}'/pth/'${data_name}
export grad_path=${result_dir}'/ideal/grads/'${model_names}'/'${data_name}'/layer_3.npy'
export lrp_path=${result_dir}'/ideal/lrps/'${model_names}'/'${data_name}'/result.npy'
export mask_path=${result_dir}'/ideal/atts/'${data_name}

python doctor/train_grad.py \
    --model_name ${model_names} \
    --data_name ${data_name} \
    --num_classes ${num_classes} \
    --num_epochs ${num_epochs} \
    --batch_size ${batch_size} \
    --origin_dir ${origin_path}\
    --result_name ${result_name} \
    --model_dir ${model_path} \
    --train_data_dir ${train_data_dir} \
    --test_data_dir ${test_data_dir} \
    --log_dir ${log_dir} \
    --grad_dir ${grad_path}\
    --lrp_dir ${lrp_path}\
    --mask_dir ${mask_path}

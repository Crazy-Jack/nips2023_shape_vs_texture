

python3 ../../train_imgnet_10.py --data_folder_path $1 \
--name resnet18_topk \
--topk 2_20 \
--total_epochs 50 \
--batch_size 32 \
--use_sin_val_folder $2 \
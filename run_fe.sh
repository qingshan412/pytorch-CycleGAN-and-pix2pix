#!/bin/bash


#################### test
FolderName=fr_lag_aug_pix2pix_transfer_b6_100_2layer
python test_fr_aligned.py \
  --dataroot ../InsightFace_Pytorch/data/facebank/noonan+normal/ \
  --name $FolderName \
  --dataset_mode unaligned \
  --model $ModelName \
  --netG resnet_4blocks \
  --num_test 100 \
  --gpu_ids 0,1 > rec/raw_${FolderName}_test 

# BatchSize=6
# ModelName=pix2pix_transfer #pix2pix #cycle_gan
# Iter=25 #25, 100, 500, 2000
# FolderName=fr_adult_${ModelName}_b${BatchSize}_${Iter}_1layer_pool5_DG_rr

# # cp -r ./checkpoints/fr_adult_basic_b6 ./checkpoints/${FolderName}
# #################### train on adults faces and then children faces
# python train_fr_aligned.py \
#   --dataroot ../InsightFace_Pytorch/data/facebank/resize+raw \
#   --continue_train \
#   --pool_size 5\
#   --name ${FolderName} \
#   --dataset_mode unaligned --model $ModelName --netG resnet_4blocks \
#   --batch_size $BatchSize --niter $Iter --niter_decay $Iter \
#   --display_id -1 --gpu_ids 0,1 \
#   --serial_batches > rec/${FolderName}_fe_rec

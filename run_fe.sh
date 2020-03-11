#!/bin/bash

BatchSize=6
ModelName=pix2pix_transfer #pix2pix #cycle_gan
Iter=2000 #25, 100, 500, 2000
FolderName=fr_adult_${ModelName}_b${BatchSize}_${Iter}_1layer_pool5

cp -r ./checkpoints/fr_adult_basic_b6 ./checkpoints/${FolderName}
#################### train on adults faces and then children faces
python train_fr_aligned.py \
  --dataroot ../InsightFace_Pytorch/data/facebank/noonan+normal \
  --continue_train \
  --pool_size 5\
  --name ${FolderName} \
  --dataset_mode unaligned --model $ModelName --netG resnet_4blocks \
  --batch_size $BatchSize --niter $Epoch --niter_decay $Epoch \
  --display_id -1 --gpu_ids 1,2 \
  --serial_batches > rec/${FolderName}_fe_rec

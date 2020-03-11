#!/bin/bash

BatchSize=6
ModelName=pix2pix_transfer #pix2pix #cycle_gan
Epoch=25 #25, 100, 500, 2000

#################### train on adults faces and then children faces
python train_fr_aligned.py \
  --dataroot ../InsightFace_Pytorch/data/facebank/noonan+normal \
  --continue_train \
  --name fr_adult_${ModelName}_b${BatchSize}_${Epoch}_1layer_fe \
  --dataset_mode unaligned --model $ModelName --netG resnet_4blocks \
  --batch_size $BatchSize --niter $Epoch --niter_decay $Epoch \
  --display_id -1 --gpu_ids 0,1 \
  --serial_batches > rec/fr_adult_${ModelName}_b${BatchSize}_${Epoch}_1layer_fe_rec 

#!/bin/bash

#$ -M jliu16@nd.edu      # Email address for job notification
#$ -m abe                # Send mail when job begins, ends and aborts
#$ -pe smp 16            # Specify parallel environment and legal core size
#$ -q gpu
#$ -l gpu_card=2
#$ -N resize          # Specify job name

module load python pytorch        # Required modules

# mkdir -p /tmp/jliu16
# rsync -a ~/Private/Research/2020/FR/InsightFace_Pytorch/data/facebank/webface /tmp/jliu16/$JOB_ID

# echo "sync success!"
# echo $CUDA_VISIBLE_DEVICES
ModelName=pix2pix_transfer #pix2pix #cycle_gan
BatchSize=6
Iter=100
#################### test
FolderName=fr_mix_aug_${ModelName}_b${BatchSize}_${Iter}_full
python test_fr_aligned.py \
  --dataroot ../InsightFace_Pytorch/data/facebank/noonan+normal/ \
  --name $FolderName \
  --dataset_mode unaligned \
  --model $ModelName \
  --netG resnet_4blocks \
  --num_test 100 \
  --gpu_ids $CUDA_VISIBLE_DEVICES > rec/${FolderName}_test 

# FolderName=fr_mix_aug_${ModelName}_b${BatchSize}_${Iter}_pool5_full
# python test_fr_aligned.py \
#   --dataroot ../InsightFace_Pytorch/data/facebank/noonan+normal/ \
#   --name $FolderName \
#   --dataset_mode unaligned \
#   --model $ModelName \
#   --netG resnet_4blocks \
#   --num_test 100 \
#   --gpu_ids $CUDA_VISIBLE_DEVICES > rec/${FolderName}_test 

# FolderName=fr_mix_aug_${ModelName}_b${BatchSize}_${Iter}_DG
# python test_fr_aligned.py \
#   --dataroot ../InsightFace_Pytorch/data/facebank/noonan+normal/ \
#   --name $FolderName \
#   --dataset_mode unaligned \
#   --model $ModelName \
#   --netG resnet_4blocks \
#   --num_test 100 \
#   --gpu_ids 0,1 > rec/${FolderName}_test 

# FolderName=fr_mix_${ModelName}_b${BatchSize}_${Iter}_DG
# python test_fr_aligned.py \
#   --dataroot ../InsightFace_Pytorch/data/facebank/noonan+normal/ \
#   --name $FolderName \
#   --dataset_mode unaligned \
#   --model $ModelName \
#   --netG resnet_4blocks \
#   --num_test 100 \
#   --gpu_ids 0,1 > rec/${FolderName}_test 

# # Epoch=20
# Iter=100 #25, 100, 500, 2000
# FolderName=fr_lag_${ModelName}_b${BatchSize}_${Iter}_2layer
# #################### train on adults faces and then children faces
# [ -d "./checkpoints/${FolderName}" ] && rm -r ./checkpoints/${FolderName}
# cp -r ./checkpoints/fr_adult_basic_b6 ./checkpoints/${FolderName}
# python train_fr_aligned.py \
#   --dataroot ../InsightFace_Pytorch/data/facebank/LAG_y_fine \
#   --continue_train \
#   --name ${FolderName} \
#   --dataset_mode unaligned --model $ModelName --netG resnet_4blocks \
#   --batch_size $BatchSize --niter $Iter --niter_decay $Iter \
#   --display_id -1 --gpu_ids $CUDA_VISIBLE_DEVICES \
#   --serial_batches > rec/${FolderName}_rec 

#################### test
# FolderName=fr_aug_pix2pix_transfer_b6_100_2layer
# python test_fr_aligned.py \
#   --dataroot ../InsightFace_Pytorch/data/facebank/noonan+normal/ \
#   --name $FolderName \
#   --dataset_mode unaligned \
#   --model $ModelName \
#   --netG resnet_4blocks \
#   --num_test 100 \
#   --gpu_ids 0,1 > rec/${FolderName}_test 

# FolderName=fr_adult_pix2pix_transfer_b6_25_2layer
# python test_fr_aligned.py \
#   --dataroot ../InsightFace_Pytorch/data/facebank/noonan+normal/ \
#   --name $FolderName \
#   --dataset_mode unaligned \
#   --model $ModelName \
#   --netG resnet_4blocks \
#   --num_test 100 \
#   --gpu_ids 0,1 > rec/${FolderName}_test 

# FolderName=fr_adult_pix2pix_transfer_b6_25_1layer_fe
# python test_fr_aligned.py \
#   --dataroot ../InsightFace_Pytorch/data/facebank/noonan+normal/ \
#   --name $FolderName \
#   --dataset_mode unaligned \
#   --model $ModelName \
#   --netG resnet_4blocks \
#   --num_test 100 \
#   --gpu_ids 0,1 > rec/${FolderName}_test 

# FolderName=fr_adult_pix2pix_transfer_b6_25_1layer_pool5_DG
# python test_fr_aligned.py \
#   --dataroot ../InsightFace_Pytorch/data/facebank/noonan+normal/ \
#   --name $FolderName \
#   --dataset_mode unaligned \
#   --model $ModelName \
#   --netG resnet_4blocks \
#   --num_test 100 \
#   --gpu_ids 0,1 > rec/${FolderName}_test


# #################### test
# FolderName=fr_adult_pix2pix_transfer_b6_25_2layer
# python test_fr_aligned.py \
#   --dataroot ../InsightFace_Pytorch/data/facebank/noonan+normal \
#   --name $FolderName \
#   --dataset_mode unaligned \
#   --model $ModelName \
#   --netG resnet_4blocks \
#   --epoch 10\
#   --num_test 100 \
#   --gpu_ids $CUDA_VISIBLE_DEVICES > rec/Dtest 

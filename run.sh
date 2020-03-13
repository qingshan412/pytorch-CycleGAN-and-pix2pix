#!/bin/bash

#$ -M jliu16@nd.edu      # Email address for job notification
#$ -m abe                # Send mail when job begins, ends and aborts
#$ -pe smp 16            # Specify parallel environment and legal core size
#$ -q gpu
#$ -l gpu_card=2
#$ -N mixed          # Specify job name

module load python pytorch        # Required modules

# mkdir -p /tmp/jliu16
# rsync -a ~/Private/Research/2020/FR/InsightFace_Pytorch/data/facebank/webface /tmp/jliu16/$JOB_ID

# echo "sync success!"
# echo $CUDA_VISIBLE_DEVICES
BatchSize=6
ModelName=pix2pix_transfer #pix2pix #cycle_gan
Epoch=500 #25, 100, 500, 2000
FolderName=fr_adult_${ModelName}_b${BatchSize}_${Epoch}_1layer_pool5_DG_mix_raw

[ -d "./checkpoints/${FolderName}" ] && rm -r ./checkpoints/${FolderName}
cp -r ./checkpoints/fr_adult_basic_b6 ./checkpoints/${FolderName}
#################### train on adults faces and then children faces
python train_fr_aligned.py \
  --dataroot ../InsightFace_Pytorch/data/facebank/webface+children \
  --continue_train \
  --pool_size 5\
  --name ${FolderName} \
  --dataset_mode unaligned --model $ModelName --netG resnet_4blocks \
  --batch_size $BatchSize --niter $Epoch --niter_decay $Epoch \
  --display_id -1 --gpu_ids $CUDA_VISIBLE_DEVICES \
  --serial_batches > rec/${FolderName}_rec 

#################### test
# python test_fr_aligned.py \
#   --dataroot ../InsightFace_Pytorch/data/facebank/noonan+normal \
#   --name fr_adult_${ModelName}_b${BatchSize} \
#   --dataset_mode unaligned \
#   --model $ModelName \
#   --netG resnet_4blocks \
#   --epoch 15\
#   --num_test 100 \
#   --gpu_ids $CUDA_VISIBLE_DEVICES > rec/Dtest 

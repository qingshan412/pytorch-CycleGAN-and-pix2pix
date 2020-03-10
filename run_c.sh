#!/bin/bash

#$ -M jliu16@nd.edu      # Email address for job notification
#$ -m abe                # Send mail when job begins, ends and aborts
#$ -pe smp 16            # Specify parallel environment and legal core size
#$ -q gpu
#$ -l gpu_card=2
#$ -N transfer          # Specify job name

module load python pytorch        # Required modules

# mkdir -p /tmp/jliu16
# rsync -a ~/Private/Research/2020/FR/InsightFace_Pytorch/data/facebank/webface /tmp/jliu16/$JOB_ID

# echo "sync success!"
# echo $CUDA_VISIBLE_DEVICES
BatchSize=6
ModelName=pix2pix_transfer #pix2pix #cycle_gan
Epoch=2000

#################### train on adults faces and then children faces
python train_fr_aligned.py \
  --dataroot ../InsightFace_Pytorch/data/facebank/noonan+normal \
  --continue_train \
  --name fr_adult_${ModelName}_b${BatchSize}_${Epoch}_2layer \
  --dataset_mode unaligned --model $ModelName --netG resnet_4blocks \
  --batch_size $BatchSize --niter $Epoch --niter_decay $Epoch \
  --display_id -1 --gpu_ids $CUDA_VISIBLE_DEVICES \
  --serial_batches > rec/fr_adult_${ModelName}_b${BatchSize}_${Epoch}_2layer_rec 
#################### test train 
# python train_fr_aligned.py \
#   --dataroot ../InsightFace_Pytorch/data/facebank/noonan+normal \
#   --continue_train \
#   --name fr_aligned_basic_b4_${ModelName}_b${BatchSize} --dataset_mode unaligned --model $ModelName \
#   --netG resnet_4blocks --batch_size $BatchSize \
#   --niter 1 --niter_decay 1 --pool_size 5\
#   --display_id -1 --gpu_ids $CUDA_VISIBLE_DEVICES --serial_batches > rec/fr_train_test_rec 
#################### test
# python test_fr_aligned.py \
#   --dataroot ../InsightFace_Pytorch/data/facebank/noonan+normal \
#   --name fr_adult_${ModelName}_b${BatchSize} \
#   --dataset_mode unaligned \
#   --model $ModelName \
#   --netG resnet_4blocks \
#   --num_test 100 \
#   --gpu_ids $CUDA_VISIBLE_DEVICES > rec/test 
#################### train on adults faces and then children faces
# python train_fr_aligned.py \
#   --dataroot ../InsightFace_Pytorch/data/facebank/noonan+normal \
#   --continue_train \
#   --name fr_aligned_basic_b4_${ModelName}_b${BatchSize} --dataset_mode unaligned --model $ModelName \
#   --netG resnet_4blocks --batch_size $BatchSize --niter 25000 --niter_decay 25000 \
#   --display_id -1 --gpu_ids $CUDA_VISIBLE_DEVICES --serial_batches > rec/fr_aligned_${ModelName}_4b_b${BatchSize}_html_serial_con_rec 
#################### train on adults faces and then children faces
# python train_fr_aligned.py \
#   --dataroot ../InsightFace_Pytorch/data/facebank/webface \
#   --name fr_adult_basic_b${BatchSize} --dataset_mode unaligned --model $ModelName \
#   --netG resnet_4blocks --batch_size $BatchSize --niter 25 --niter_decay 25 \
#   --display_id -1 --gpu_ids $CUDA_VISIBLE_DEVICES --serial_batches > rec/fr_adult_basic_b${BatchSize}

# /bin/rm -r /tmp/jliu16/$JOB_ID



# #####################################N New data, complete multi-cycle
# python train_ct_3.py --dataroot ./datasets/june_13_2 --name june13_2_MC_batch2 --dataset_mode unalignedC --model cycle_gan_c --netG resnet_4blocks --batch_size 2 --niter 25 --niter_decay 25 --display_id -1 --gpu_ids 0 > rec/june13_2_MC_batch2_html_rec &
# python test_ct_4.py --dataroot ./datasets/june_13_2 --name june13_2_MC_batch2 --dataset_mode unalignedC --model cycle_gan_c --netG resnet_4blocks --serial_batches --no_flip

# #####################################N New data, orignal cycle-gan
# python train_ct_3.py --dataroot ./datasets/june_13_2 --name june13_2_OC_batch2 --dataset_mode unalignedCT --model cycle_gan --netG resnet_9blocks --batch_size 2 --niter 25 --niter_decay 25 --display_id -1 --gpu_ids 0,1 > rec/june13_2_OC_batch2_html_rec &
# python test_ct_4.py --dataroot ./datasets/june_13_2 --name june13_2_OC_batch2 --dataset_mode unalignedCT --model cycle_gan --netG resnet_9blocks --serial_batches --no_flip

# #####################################N New data, only local cycles
# python train_ct_3.py --dataroot ./datasets/june_13_2_lc_ac --name june13_2_LC_ac_batch2 --dataset_mode unalignedCT --model cycle_gan --netG resnet_4blocks --batch_size 2 --niter 25 --niter_decay 25 --display_id -1 --gpu_ids 0 > rec/june13_2_LC_ac_batch2_html_rec &
# python train_ct_3.py --dataroot ./datasets/june_13_2_lc_cb --name june13_2_LC_cb_batch2 --dataset_mode unalignedCT --model cycle_gan --netG resnet_4blocks --batch_size 2 --niter 25 --niter_decay 25 --display_id -1 --gpu_ids 1 > rec/june13_2_LC_cb_batch2_html_rec &
# python test_ct_4.py --dataroot ./datasets/june_13_2 --name june13_2_LC_batch2 --dataset_mode unalignedC --model cycle_gan_cd --netG resnet_4blocks --serial_batches --no_flip

# #####################################N New data, only global cycles
# python train_ct_3.py --dataroot ./datasets/june_13_2 --name june13_2_GC_batch2 --dataset_mode unalignedC --model cycle_gan_cl --netG resnet_4blocks --batch_size 2 --niter 25 --niter_decay 25 --display_id -1 --gpu_ids 1 > rec/june13_2_GC_batch2_html_rec &
# python test_ct_4.py --dataroot ./datasets/june_13_2 --name june13_2_GC_batch2 --dataset_mode unalignedC --model cycle_gan_cl --netG resnet_4blocks --serial_batches --no_flip


#####################################N New data, original cycle-gan
# python train_ct_3.py --dataroot ./datasets/june_13_1 --name june13_1_OC_batch2 --dataset_mode unalignedC --model cycle_gan --netG resnet_9blocks --batch_size 2 --niter 25 --niter_decay 25 --display_id -1 --gpu_ids 0,1 > rec/june13_1_OC_batch2_html_rec &
# python test_ct_4.py --dataroot ./datasets/june_13_1 --name june13_1_OC_batch2 --dataset_mode unalignedC --model cycle_gan --netG resnet_9blocks --serial_batches

#####################################complete MCCAN with local cycles of fake C
# python3 train_ct_3.py --dataroot ./datasets/tw_np_200 --name twnp200c_cyclegan4cc_batch2 --dataset_mode unalignedC --model cycle_gan_cc --netG resnet_4blocks --batch_size 2 --niter 25 --niter_decay 25 --display_id -1 --gpu_ids 1 > rec/twnp200c_cyclegan4cc_batch2_html_rec &
# python3 test_ct_3.py --dataroot ./datasets/tw_np_200 --name twnp200c_cyclegan4cc_batch2 --dataset_mode unalignedC --model cycle_gan_cc --netG resnet_4blocks --serial_batches

#####################################C_test
# python train_ct_3.py --dataroot ./datasets/tw_np_all --name twnpac_cycleganc_batch4 --dataset_mode unalignedC --model cycle_gan_c --batch_size 4 --niter 25 --niter_decay 25 --display_id -1 > rec/twnpac_cycleganc_batch4_html_rec &

# python train_ct_3.py --dataroot ./datasets/tw_np_200 --name twnp200c_cycleganc_batch2 --dataset_mode unalignedC --model cycle_gan_c --batch_size 2 --niter 25 --niter_decay 25 --display_id -1 --gpu_ids 1 > rec/twnp200c_cycleganc_batch2_html_rec &
# python test_ct_3.py --dataroot ./datasets/tw_np_200 --name twnp200c_cycleganc_batch2 --dataset_mode unalignedC --model cycle_gan_c --serial_batches

# python test_ct_3.py --dataroot ./datasets/tw_np_200 --name twnp200c_cyclegan6c_batch2 --dataset_mode unalignedC --model cycle_gan_c --netG resnet_6blocks --serial_batches
# python train_ct_3.py --dataroot ./datasets/tw_np_200 --name twnp200c_cyclegan6c_batch2 --dataset_mode unalignedC --model cycle_gan_c --netG resnet_6blocks --batch_size 2 --niter 25 --niter_decay 25 --display_id -1 --gpu_ids 1 > rec/twnp200c_cyclegan6c_batch2_html_rec &

# python test_ct_3.py --dataroot ./datasets/tw_np_200 --name twnp200c_cyclegan4c_batch2 --dataset_mode unalignedC --model cycle_gan_c --netG resnet_4blocks --serial_batches
# python train_ct_3.py --dataroot ./datasets/tw_np_200 --name twnp200c_cyclegan6c_batch2 --dataset_mode unalignedC --model cycle_gan_c --netG resnet_4blocks --batch_size 2 --niter 25 --niter_decay 25 --display_id -1 --gpu_ids 1 > rec/twnp200c_cyclegan6c_batch2_html_rec &
#####################################Original
# python train_ct.py --dataroot ./datasets/tw_np_200 --name twnp200_cyclegan_idt0_batch4 --model cycle_gan --batch_size 4 --niter 25 --niter_decay 25 --display_id -1 --gpu_ids 1 > rec/twnp200_cyclegan_idt0_batch4_html_rec
# python train_ct.py --dataroot ./datasets/tw_np_200 --name twnp200_cyclegan_batch4 --model cycle_gan --batch_size 4 --niter 25 --niter_decay 25 --display_id -1 --gpu_ids 1 > rec/twnp200_cyclegan_batch4_html_rec 
# python test_ct_2.py --dataroot ./datasets/tw_np_200 --name twnp200_cyclegan_batch4 --model cycle_gan --serial_batches
# python train_ct.py --dataroot ./datasets/tw_np_200 --name twnp200_cyclegan_batch2 --model cycle_gan --batch_size 2 --niter 25 --niter_decay 25 --display_id -1 --gpu_ids 1 > rec/twnp200_cyclegan_batch2_html_rec 
# python train_ct.py --dataroot ./datasets/tw_np_200 --name twnp200_cyclegan_iter50_batch2 --model cycle_gan --batch_size 2 --niter 50 --niter_decay 50 --display_id -1 --gpu_ids 1 > rec/twnp200_cyclegan_iter50_batch2_html_rec
# python train_ct.py --dataroot ./datasets/tw_np_200 --name twnp200_cyclegan_idt0_batch4 --model cycle_gan --batch_size 4 --niter 25 --niter_decay 25 --display_id -1 --gpu_ids 1 > rec/twnp200_cyclegan_idt0_batch4_html_rec
# python test_ct_2.py --dataroot ./datasets/tw_np_200 --name twnp200_cyclegan_iter50_batch2 --model cycle_gan --serial_batches
#####################################Fake_train
# python ProdFake.py --dataroot ./datasets/tw_np_200_fake --name twnp200_cyclegan_batch4 --model testCT --serial_batches
# python train_ct.py --dataroot ./datasets/tw_np_200_fake --name twnp200fake_cyclegan_batch4 --model cycle_gan --batch_size 4 --niter 25 --niter_decay 25 --display_id -1 --gpu_ids 1 > rec/twnp200fake_cyclegan_batch4_html_rec 
# python test_ct_2.py --dataroot ./datasets/tw_np_200_fake --name twnp200fake_cyclegan_batch4 --model testCT --serial_batches

#####################################One_large_cycle
# python train_ct_3.py --dataroot ./datasets/tw_np_200 --name twnp200c_cyclegancl_batch2 --dataset_mode unalignedC --model cycle_gan_cl --batch_size 2 --niter 25 --niter_decay 25 --display_id -1 > rec/twnp200c_cyclegancl_batch2_html_rec &
# python test_ct_3.py --dataroot ./datasets/tw_np_200 --name twnp200c_cyclegancl_batch2 --dataset_mode unalignedC --model cycle_gan_cl --serial_batches
# python train_ct_3.py --dataroot ./datasets/tw_np_200 --name new_twnp200c_cyclegan4cl_batch2 --dataset_mode unalignedC --netG resnet_4blocks --model cycle_gan_cl --batch_size 2 --niter 25 --niter_decay 25 --display_id -1 > rec/new_twnp200c_cyclegan4cl_batch2_html_rec &
# python train_ct_3.py --dataroot ./datasets/tw_np_200 --name new_twnp200c_cyclegan4cl_iter50_batch2 --dataset_mode unalignedC --netG resnet_4blocks --model cycle_gan_cl --batch_size 2 --niter 50 --niter_decay 50 --display_id -1 --gpu_ids 1 > rec/new_twnp200c_cyclegan4cl_iter50_batch2_html_rec &
# python test_ct_3.py --dataroot ./datasets/tw_np_200 --name new_twnp200c_cyclegan4cl_batch2 --dataset_mode unalignedC --model cycle_gan_cl --netG resnet_4blocks --serial_batches
# python test_ct_3.py --dataroot ./datasets/tw_np_200 --name new_twnp200c_cyclegan4cl_iter50_batch2 --dataset_mode unalignedC --model cycle_gan_cl --netG resnet_4blocks --serial_batches


# python train_ct_3.py --dataroot ./datasets/tw_np_200 --name new_twnp200c_cyclegan4cl2_batch2 --dataset_mode unalignedC --netG resnet_4blocks --model cycle_gan_cl2 --batch_size 2 --niter 25 --niter_decay 25 --display_id -1 > rec/new_twnp200c_cyclegan4cl2_batch2_html_rec &

# python train_ct.py --dataroot ./datasets/tw_np_200 --name twnp200_cyclegan_batch4 --model cycle_gan --batch_size 4 --niter 25 --niter_decay 25 --display_id -1 --gpu_ids 1 > rec/twnp200_cyclegan_batch4_html_rec
#####################################C_decouple
# python train_ct.py --dataroot ./datasets/decouple_AC200 --name decouple_ac200_cyclegan_batch4 --model cycle_gan --batch_size 4 --niter 25 --niter_decay 25 --display_id -1 --gpu_ids 1 > rec/decouple_ac200_cyclegan_batch4_html_rec
# python train_ct.py --dataroot ./datasets/decouple_CB200 --name decouple_cb200_cyclegan_batch4 --model cycle_gan --batch_size 4 --niter 25 --niter_decay 25 --display_id -1 --gpu_ids 1 > rec/decouple_cb200_cyclegan_batch4_html_rec
# python test_ct_2.py --dataroot ./datasets/decouple_AC200 --name decouple_ac200_cyclegan_batch4 --model cycle_gan --serial_batches
# python test_ct_2.py --dataroot ./datasets/decouple_CB200 --name decouple_cb200_cyclegan_batch4 --model cycle_gan --serial_batches

# python train_ct.py --dataroot ./datasets/decouple_AC200 --name decouple_ac200_cyclegan4_iter50_batch2 --model cycle_gan --netG resnet_4blocks --batch_size 2 --niter 50 --niter_decay 50 --display_id -1 > rec/decouple_ac200_cyclegan_iter50_batch2_html_rec
# python train_ct.py --dataroot ./datasets/decouple_CB200 --name decouple_cb200_cyclegan4_iter50_batch2 --model cycle_gan --netG resnet_4blocks --batch_size 2 --niter 50 --niter_decay 50 --display_id -1 > rec/decouple_cb200_cyclegan_iter50_batch2_html_rec
# python test_ct_2.py --dataroot ./datasets/decouple_AC200 --name decouple_ac200_cyclegan4_iter50_batch2 --model cycle_gan --netG resnet_4blocks --serial_batches
# python test_ct_2.py --dataroot ./datasets/decouple_CB200 --name decouple_cb200_cyclegan4_iter50_batch2 --model cycle_gan --netG resnet_4blocks --serial_batches
# python test_ct_2.py --dataroot ./datasets/decouple_CB200 --name decouple_cb200_cyclegan_batch2 --model cycle_gan --serial_batches

# python train_ct.py --dataroot ./datasets/decouple_AC200 --name decouple_ac200_cyclegan_batch4 --model cycle_gan --batch_size 4 --niter 25 --niter_decay 25 --display_id -1 --gpu_ids 1 > rec/decouple_ac200_cyclegan_batch4_html_rec
# python train_ct.py --dataroot ./datasets/decouple_CD200 --name decouple_cd200_cyclegan_batch4 --model cycle_gan --batch_size 4 --niter 25 --niter_decay 25 --display_id -1 --gpu_ids 1 > rec/decouple_cd200_cyclegan_batch4_html_rec
# python train_ct.py --dataroot ./datasets/decouple_DB200 --name decouple_db200_cyclegan_batch4 --model cycle_gan --batch_size 4 --niter 25 --niter_decay 25 --display_id -1 > rec/decouple_db200_cyclegan_batch4_html_rec
# python test_ct_2.py --dataroot ./datasets/decouple_CD200 --name decouple_cd200_cyclegan_batch4 --model cycle_gan --serial_batches
# python test_ct_2.py --dataroot ./datasets/decouple_DB200 --name decouple_db200_cyclegan_batch4 --model testCT --serial_batches

#####################################npy
### partial tw dataset
# python train_ct.py --dataroot ./datasets/tw_np --name twnp_cyclegan_batch4 --model cycle_gan --batch_size 4 --niter 25 --niter_decay 25 --display_id -1 > rec/twnp_cyclegan_batch4_html_rec

### all tw dataset
# python train_ct.py --dataroot ./datasets/tw_np_all --name twnpa_cyclegan_batch4 --model cycle_gan --batch_size 4 --niter 25 --niter_decay 25 --display_id -1 --gpu_ids 1 > rec/twnpa_cyclegan_batch4_html_rec
# python test_ct_1.py --dataroot ./datasets/tw_np_all --name twnpa_cyclegan_batch4 --model cycle_gan --serial_batches

### all tw dataset filtering test out + active learning
# python train_al.py --dataroot ./datasets/tw_np_all --name twnpa_cyclegan_batch4 --model cycle_gan --batch_size 4 --niter 25 --niter_decay 25 --display_id -1 --gpu_ids 1 > rec/twnpa_cyclegan_batch4_html_rec
# python test_ct_al.py --dataroot ./datasets/tw_np_all --name twnpa_cyclegan_batch4 --model cycle_gan --serial_batches

### one tw dataset
# python train_ct.py --dataroot ./datasets/tw_np_one --name twnpo_cyclegan_batch4 --model cycle_gan --batch_size 4 --niter 25 --niter_decay 25 --display_id -1 --gpu_ids 1 > rec/twnpo_cyclegan_batch4_html_rec
# python test_ct_1.py --dataroot ./datasets/tw_np_one --name twnpo_cyclegan_batch4 --model cycle_gan
# python train_ct.py --dataroot ./datasets/tw_np_one --name twnpo_cyclegan_batch4_epoch100 --model cycle_gan --batch_size 4 --niter 100 --niter_decay 100 --display_id -1 --gpu_ids 1 > rec/twnpo_cyclegan_batch4_epoch100_html_rec
# python test_ct_1.py --dataroot ./datasets/tw_np_one --name twnpo_cyclegan_batch4_epoch100 --model cycle_gan --serial_batches

### one/two tw dataset
# python train_ct.py --dataroot ./datasets/tw_np_one --name twnpot_cyclegan_batch4 --model cycle_gan --batch_size 4 --niter 25 --niter_decay 25 --display_id -1 > rec/twnpot_cyclegan_batch4_html_rec
# nohup python train_ct.py --dataroot ./datasets/tw_np_one --name twnpot_cyclegan_batch4 --model cycle_gan --batch_size 4 --niter 25 --niter_decay 25 --display_id -1 > rec/twnpot_cyclegan_batch4_html_rec &
# python test_ct_2.py --dataroot ./datasets/tw_np_one --name twnpot_cyclegan_batch4 --model cycle_gan --serial_batches

### one/partially two dataset
# python train_ct.py --dataroot ./datasets/tw_np_one --name twnpopt_cyclegan_batch4 --model cycle_gan --batch_size 4 --niter 25 --niter_decay 25 --display_id -1 --gpu_ids 1 > rec/twnpopt_cyclegan_batch4_html_rec
# nohup python train_ct.py --dataroot ./datasets/tw_np_one --name twnpot_cyclegan_batch4 --model cycle_gan --batch_size 4 --niter 25 --niter_decay 25 --display_id -1 > rec/twnpot_cyclegan_batch4_html_rec &
# python test_ct_2.py --dataroot ./datasets/tw_np_one --name twnpot_cyclegan_batch4 --model cycle_gan --serial_batches

### regenerated one dataset
# python train_ct.py --dataroot ./datasets/tw_np_one_1 --name twnpo1_cyclegan_batch4 --model cycle_gan --batch_size 4 --niter 25 --niter_decay 25 --display_id -1 --gpu_ids 1 > rec/twnpo1_cyclegan_batch4_html_rec
# python test_ct_2.py --dataroot ./datasets/tw_np_one --name twnpo1_cyclegan_batch4 --model cycle_gan --serial_batches

#####################################dcm
### batch4
# python train.py --dataroot ./datasets/ctest --name ctest_cyclegan_batch4 --model cycle_gan --batch_size 4 --niter 25 --niter_decay 25 --display_id -1 --no_html > rec/ctest_cyclegan_batch4_rec
# python test_ct_1.py --dataroot ./datasets/ctest --name ctest_cyclegan_batch4 --model cycle_gan

# python train_ct.py --dataroot ./datasets/ctest --name ctest_cyclegan_batch4 --model cycle_gan --batch_size 4 --niter 25 --niter_decay 25 --display_id -1 > rec/ctest_cyclegan_batch4_html_rec
# python train_ct.py --dataroot ./datasets/ctest --name ctest_cyclegan_batch4_epoch50 --model cycle_gan --batch_size 4 --niter 50 --niter_decay 50 --display_id -1 > rec/ctest_cyclegan_batch4_epoch50_html_rec
# python train_ct.py --dataroot ./datasets/ctest --name ctest_cyclegan_batch4_epoch50_idt0 --model cycle_gan --batch_size 4 --niter 50 --niter_decay 50 --display_id -1 --gpu_ids 1 --lambda_identity 0 > rec/ctest_cyclegan_batch4_epoch50_idt0_html_rec
# python train_ct.py --dataroot ./datasets/ctest --name ctest_cyclegan_batch4_epoch50_idt0_AB5 --model cycle_gan --batch_size 4 --niter 50 --niter_decay 50 --display_id -1 --lambda_identity 0 --lambda_A 5.0 --lambda_B 5.0 > rec/ctest_cyclegan_batch4_epoch50_idt0_AB5_html_rec

### low
# python train_ct.py --dataroot ./datasets/ctestL --name ctestL_cyclegan_batch4 --model cycle_gan --batch_size 4 --niter 25 --niter_decay 25 --display_id -1 > rec/ctestL_cyclegan_batch4_html_rec

### batch2
# python train.py --dataroot ./datasets/ctest --name ctest_cyclegan_batch2 --model cycle_gan --batch_size 2 --niter 50 --niter_decay 50 --display_id -1 --no_html > rec/ctest_cyclegan_batch2_rec
# python test.py --dataroot ./datasets/ctest --name ctest_cyclegan_batch2 --model cycle_gan > rec/ctest_cyclegan_batch2_test_rec

### original samples
# rm -r ./checkpoints/maps_cyclegan
# python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan --display_id -1 --no_html --gpu_ids 1 > rec/maps_cyclegan_rec
# python train.py --dataroot ./datasets/maps --name maps_cyclegan_batch2 --model cycle_gan --batch_size 2 --niter 50 --niter_decay 50 --display_id -1 --no_html > rec/maps_cyclegan_batch2_rec
# python train.py --dataroot ./datasetss/maps --name maps_cyclegan_batch20_iter10 --model cycle_gan --batch_size 20 --niter 10 --niter_decay 10 --display_id -1 --no_html > rec/maps_cyclegan_batch20_iter10_rec
# python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
# python test.py --dataroot ./datasets/horse2zebra/testA --name horse2zebra_pretrained --model test

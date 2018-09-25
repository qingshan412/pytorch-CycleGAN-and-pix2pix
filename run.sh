python train.py --dataroot ./datasets/ctest --name ctest_cyclegan_batch4 --model cycle_gan --batch_size 4 --niter 25 --niter_decay 25 --display_id -1 --no_html > rec/ctest_cyclegan_batch4_rec
# python train.py --dataroot ./datasets/ctest --name ctest_cyclegan_batch2 --model cycle_gan --batch_size 2 --niter 50 --niter_decay 50 --display_id -1 --no_html > rec/ctest_cyclegan_batch2_rec

# rm -r ./checkpoints/maps_cyclegan
# python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan --display_id -1 --no_html --gpu_ids 1 > rec/maps_cyclegan_rec
# python train.py --dataroot ./datasets/maps --name maps_cyclegan_batch2 --model cycle_gan --batch_size 2 --niter 50 --niter_decay 50 --display_id -1 --no_html > rec/maps_cyclegan_batch2_rec
# python train.py --dataroot ./datasetss/maps --name maps_cyclegan_batch20_iter10 --model cycle_gan --batch_size 20 --niter 10 --niter_decay 10 --display_id -1 --no_html > rec/maps_cyclegan_batch20_iter10_rec
# python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
# python test.py --dataroot ./datasets/horse2zebra/testA --name horse2zebra_pretrained --model test
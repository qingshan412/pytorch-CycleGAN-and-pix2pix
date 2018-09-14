# rm -r ./checkpoints/maps_cyclegan
# python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan --display_id -1 --no_html > rec/maps_cyclegan_rec

python train.py --dataroot ./datasetss/maps --name maps_cyclegan_batch20 --model cycle_gan --batch_size 20 --niter 5 --niter_decay 5 --display_id -1 --no_html > rec/maps_cyclegan_batch20_rec

# python train.py --dataroot ./datasetss/maps --name maps_cyclegan_batch20_iter10 --model cycle_gan --batch_size 20 --niter 10 --niter_decay 10 --display_id -1 --no_html > rec/maps_cyclegan_batch20_iter10_rec
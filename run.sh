rm -r ./checkpoints/maps_cyclegan
python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan --display_id -1 --no_html > rec/maps_cyclegan_rec
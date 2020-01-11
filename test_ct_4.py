'''
Merge new test images into one image.
'''
import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_ct_npy, save_merged_npy
from util import html


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # no shuffle
    # opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display
    # opt.num_test = 20 
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    model.setup(opt)
    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch)) ### ./results/ctest_cyclegan/test_[epoch]
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test
    need_merge = {}
    for i, data in enumerate(dataset):
        if i >= opt.num_test:
            break
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        print('processing (%04d)-th image... %s' % (i, img_path))
        if 'CT1_' in img_path[0] or 'low_dose_data_' in img_path[0]:
            merged_name = img_path[0].split('/')[-1].split('.')[0][:-2]
            merge_num = int(img_path[0].split('/')[-1].split('.')[0][-1])
            if merged_name not in need_merge:
                need_merge[merged_name] = [None, None, None, None]
            need_merge[merged_name][merge_num] = visuals
        else:
            save_ct_npy(webpage, visuals, img_path, width=opt.display_winsize)
    
    MergedDisplayWinSize = 512
    save_merged_npy(webpage, need_merge, width = MergedDisplayWinSize)

    webpage.save()

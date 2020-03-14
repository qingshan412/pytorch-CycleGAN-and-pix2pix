import numpy as np
import os
import sys
import ntpath
import time
from . import util
from . import html
from PIL import Image
# from scipy.misc import imresize

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError

def imresize(arr, width, height):
    return np.array(Image.fromarray(arr).resize((width, height), Image.BICUBIC))

# save image to the disk
def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    image_dir = webpage.get_image_dir() ### ./results/ctest_cyclegan/test_[epoch]
    short_path = ntpath.basename(image_path[0]) ### testA1.dcm
    name = os.path.splitext(short_path)[0] ### testA1

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        h, w, _ = im.shape
        if aspect_ratio > 1.0:
            im = imresize(im, int(w * aspect_ratio), h)
            # im = imresize(im, (h, int(w * aspect_ratio)), interp='bicubic')
        if aspect_ratio < 1.0:
            im = imresize(im, w, int(h / aspect_ratio))
            # im = imresize(im, (int(h / aspect_ratio), w), interp='bicubic')
        util.save_image(im, save_path)

        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)

# save ct numpy arrays to npy
def save_ct2npy(web_dir, visuals, image_path):
    #webpage, , width=256
    # if not os.path.exists(web_dir):
        # os.makedirs(web_dir)
    image_dir = os.path.join(web_dir, 'npys')#webpage.get_image_dir() ### ./results/ctest_cyclegan/test_[epoch]
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    short_path = ntpath.basename(image_path[0]) ### testA1.dcm
    name = os.path.splitext(short_path)[0] ### testA1

    # webpage.add_header(name)
    # ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        im_numpy = np.squeeze(util.tensor2ctim(im_data))
        # image_name = '%s_%s.png' % (name, label)
        # save_path = os.path.join(image_dir, image_name)
        # _, h, w = im.shape
        # if aspect_ratio > 1.0:
        #     im = imresize(im, (h, int(w * aspect_ratio)), interp='bicubic')
        # if aspect_ratio < 1.0:
        #     im = imresize(im, (int(h / aspect_ratio), w), interp='bicubic')
        np.save(os.path.join(image_dir, '%s_%s.npy' % (name, label)), im_numpy)
        # if name.split('+')[0] != "fbp":
        #     mean_str, std_str = util.save_ctABo_image(im_numpy, save_path)
        # else:
        #     if label in ['real_A', 'fake_B', 'rec_A', 'idt_B', 'fake_B_A', 'rec_A_B', 'fake_C_A', 'rec_A_C']:
        #         mean_str, std_str = util.save_ctA_image(im_numpy, save_path)
        #     else:
        #         mean_str, std_str = util.save_ctB_image(im_numpy, save_path)
        

        # util.save_cti_image(im_numpy, save_path, label)
    #     ims.append(image_name)
    #     txts.append(label + ', mean:' + ', '.join(mean_str) + ', std:' + ', '.join(std_str))
    #     links.append(image_name)
    # webpage.add_images(ims, txts, links, width=width)

# save ct numpy arrays to npy, add to a webpage
def save_ct_npy(webpage, visuals, image_path, width=256):
    image_dir = webpage.get_image_dir() ### ./results/ctest_cyclegan/test_[epoch]
    short_path = ntpath.basename(image_path[0]) ### testA1.dcm
    name = os.path.splitext(short_path)[0] ### testA1

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        im_numpy = np.squeeze(util.tensor2ctim(im_data))
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        # _, h, w = im.shape
        # if aspect_ratio > 1.0:
        #     im = imresize(im, (h, int(w * aspect_ratio)), interp='bicubic')
        # if aspect_ratio < 1.0:
        #     im = imresize(im, (int(h / aspect_ratio), w), interp='bicubic')
        
        np.save(os.path.join(image_dir, '%s_%s.npy' % (name, label)), im_numpy)
        mean_str, std_str = util.save_ct_image(name, label, im_numpy, save_path)
        # if name.split('+')[0] != "fbp":
        #     if name.split('_')[0] != "199" and name.split('_')[0] != "200":
        #         mean_str, std_str = util.save_ct_image(im_numpy, save_path)
        #     else:
        #         mean_str, std_str = util.save_ctABo_image(im_numpy, save_path)
        # else:
        #     if label in ['real_A', 'fake_B', 'rec_A', 'idt_B', 'fake_B_A', 'rec_A_B', 'fake_C_A', 'rec_A_C']:
        #         mean_str, std_str = util.save_ctA_image(im_numpy, save_path)
        #     else:
        #         mean_str, std_str = util.save_ctB_image(im_numpy, save_path)
        

        # util.save_cti_image(im_numpy, save_path, label)
        ims.append(image_name)
        txts.append(label + ', mean:' + ', '.join(mean_str) + ', std:' + ', '.join(std_str))
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)


def save_merged_npy(webpage, NeedMerge, width=512):
    image_dir = webpage.get_image_dir()
    for name, attrs in NeedMerge.items():
        if None in attrs:
            print('Some parts missing in ' + name + '!')
            continue
        webpage.add_header(name)
        ims, txts, links = [], [], []

        ### Get the original pixel average
        part_0 = np.squeeze(util.tensor2ctim(attrs[0]['real_A']))
        part_1 = np.squeeze(util.tensor2ctim(attrs[1]['real_A']))
        part_2 = np.squeeze(util.tensor2ctim(attrs[2]['real_A']))
        part_3 = np.squeeze(util.tensor2ctim(attrs[3]['real_A']))
        pixel_avg_real_A = np.mean(np.squeeze(np.concatenate((
            np.concatenate((part_0, part_1), axis = 1), 
            np.concatenate((part_2, part_3), axis = 1)
            ), axis = 0)))

        for label, _ in attrs[0].items():
            # Merge images back
            # 0,1
            # 2,3 -merge-> merged_visual
            part_0 = np.squeeze(util.tensor2ctim(attrs[0][label]))
            part_1 = np.squeeze(util.tensor2ctim(attrs[1][label]))
            part_2 = np.squeeze(util.tensor2ctim(attrs[2][label]))
            part_3 = np.squeeze(util.tensor2ctim(attrs[3][label]))
            im_numpy = np.squeeze(np.concatenate((
                np.concatenate((part_0, part_1), axis = 1), 
                np.concatenate((part_2, part_3), axis = 1)
                ), axis = 0))
            # adjustment according to pixel_avg_real_A
            im_numpy = im_numpy + (pixel_avg_real_A - np.mean(im_numpy))
            im_numpy[np.where(im_numpy > 4095)] = 4095.
            im_numpy[np.where(im_numpy < 0)] = 0.
            # For webpage display
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            np.save(os.path.join(image_dir, '%s_%s.npy' % (name, label)), im_numpy)
            mean_str, std_str = util.save_merged_ct_image_8(im_numpy, save_path)            

            ims.append(image_name)
            txts.append(label + ', mean:' + ', '.join(mean_str) + 
                ', std:' + ', '.join(std_str))
            links.append(image_name)
        
        webpage.add_images(ims, txts, links, width=width)


class Visualizer():
    def __init__(self, opt):
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.opt = opt
        self.saved = False
        self.epochs = [] # epoch to save and display
        if self.display_id > 0:
            import visdom
            self.ncols = opt.display_ncols
            self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, env=opt.display_env, raise_exceptions=True)

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        self.saved = False

    def throw_visdom_connection_error(self):
        print('\n\nCould not connect to Visdom server (https://github.com/facebookresearch/visdom) for displaying training progress.\nYou can suppress connection to Visdom using the option --display_id -1. To install visdom, run \n$ pip install visdom\n, and start the server by \n$ python -m visdom.server.\n\n')
        exit(1)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, save_result):
        if self.display_id > 0:  # show images in the browser
            ncols = self.ncols
            if ncols > 0:
                ncols = min(ncols, len(visuals))
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
                        table {border-collapse: separate; border-spacing:4px; white-space:nowrap; text-align:center}
                        table td {width: %dpx; height: %dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)
                title = self.name
                label_html = ''
                label_html_row = ''
                images = []
                idx = 0
                for label, image in visuals.items():
                    image_numpy = util.tensor2im(image)
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                # pane col = image row
                try:
                    self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                    padding=2, opts=dict(title=title + ' images'))
                    label_html = '<table>%s</table>' % label_html
                    self.vis.text(table_css + label_html, win=self.display_id + 2,
                                  opts=dict(title=title + ' labels'))
                except VisdomExceptionBase:
                    self.throw_visdom_connection_error()

            else:
                idx = 1
                for label, image in visuals.items():
                    image_numpy = util.tensor2im(image)
                    self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                   win=self.display_id + idx)
                    idx += 1
        
        if self.use_html and (save_result or not self.saved):  # save images to a html file
            self.saved = True
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)
            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, reflesh=1)
            if not self.epochs:
                self.epochs.append(epoch)
            else:
                if self.epochs[-1] != epoch:
                    self.epochs.append(epoch)
            for i in range(len(self.epochs), 0, -1):
                n = self.epochs[i-1]
                webpage.add_header('epoch [%d]' % n)
                ims, txts, links = [], [], []

                for label, image_numpy in visuals.items():
                    image_numpy = util.tensor2im(image)
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    def display_current_ct_results(self, visuals, epoch, save_result):
        if self.display_id > 0:  # show images in the browser
            ncols = self.ncols
            if ncols > 0:
                ncols = min(ncols, len(visuals))
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
                        table {border-collapse: separate; border-spacing:4px; white-space:nowrap; text-align:center}
                        table td {width: %dpx; height: %dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)
                title = self.name
                label_html = ''
                label_html_row = ''
                images = []
                idx = 0
                for label, image in visuals.items():
                    image_numpy = util.tensor2im(image)
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                # pane col = image row
                try:
                    self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                    padding=2, opts=dict(title=title + ' images'))
                    label_html = '<table>%s</table>' % label_html
                    self.vis.text(table_css + label_html, win=self.display_id + 2,
                                  opts=dict(title=title + ' labels'))
                except VisdomExceptionBase:
                    self.throw_visdom_connection_error()

            else:
                idx = 1
                for label, image in visuals.items():
                    image_numpy = util.tensor2im(image)
                    self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                   win=self.display_id + idx)
                    idx += 1

        if self.use_html and (save_result or not self.saved):  # save images to a html file
            self.saved = True
            for label, image in visuals.items():
                image_numpy = util.tensor2ctim(image)
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_ct_image_naive(image_numpy, img_path)
            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, reflesh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims, txts, links = [], [], []

                for label, image_numpy in visuals.items():
                    # image_numpy = util.tensor2ctim(image)
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    # losses: dictionary of error labels and values
    def plot_current_losses(self, epoch, counter_ratio, opt, losses):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
                Y=np.array(self.plot_data['Y']),
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss',
                    'width': 800},
                win=self.display_id)
        except VisdomExceptionBase:
            self.throw_visdom_connection_error()

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, i, losses, t, t_data):
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, i, t, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

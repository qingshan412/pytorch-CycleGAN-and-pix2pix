from __future__ import division
from PIL import Image
from os import listdir, path, makedirs
#from os.path import isfile, join, exist
import numpy as np
import re
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument("-s","--SourceDir", type=str,
                    help="directory for test raw samples",
                    default = 'maps_cyclegan')

args = parser.parse_args()
LogPath = path.join('./checkpoints', args.SourceDir, 'loss_log.txt')

# Files = [ f for f in listdir(PicPath) if path.isfile(path.join(LogPath,f))]

# for Fic in Files:
idx_name = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
losses = {}
for i in range(len(idx_name)):
    losses[idx_name[i]] = []

print('Reading from ' + LogPath + '...')
fo = open(LogPath)
while 1:
    lines = fo.readlines(10000)
    if not lines:
        break
    for line in lines:
        line = line.strip().split(' ')
        if len(line) > 20:
            for i in range(len(idx_name)):
                losses[idx_name[i]].append(float(line[2*i+9]))

plt.figure()
for i in range(len(idx_name)):
    plt.plot(losses[idx_name[i]], label=idx_name[i])
# plt.plot(d_loss, 'ro', label='d_loss')
# plt.plot(g_loss, 'go', label='g_loss')
plt.legend()
plt.title(args.SourceDir)
plt.show()
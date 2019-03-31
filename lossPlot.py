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

parser.add_argument("-n","--Normalized", 
                    help="Use normalized loss values",
                    default=False, action='store_true')

parser.add_argument("-c","--CycleLossOnly", 
                    help="Plot cycle loss only",
                    default=False, action='store_true')

args = parser.parse_args()
LogPath = path.join('./checkpoints', args.SourceDir, 'loss_log.txt')

# Files = [ f for f in listdir(PicPath) if path.isfile(path.join(LogPath,f))]

# for Fic in Files:
# idx_name = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
idx_name = []
losses = {}

print('Reading from ' + LogPath + '...')
fo = open(LogPath)
flag = 1
while 1:
    lines = fo.readlines(10000)
    if not lines:
        break
    for line in lines:
        line = line.strip().split(' ')
        if len(line) > 20:
            if flag:
                i = 8
                while i < len(line):
                    if 'decouple_ac200' in args.SourceDir:
                        if line[i][0].upper == 'D' or line[i][0].upper == 'G':
                            name_i = line[i][0].upper() + line[i][1:-1].replace('A', 'Z').replace('B', 'X')
                        else:
                            name_i = line[i][0].upper() + line[i][1:-1].replace('A', 'X').replace('B', 'Z')
                    elif 'decouple_cb200' in args.SourceDir:
                        if line[i][0].upper == 'D' or line[i][0].upper == 'G':
                            name_i = line[i][0].upper() + line[i][1:-1].replace('A', 'Y').replace('B', 'Z')
                        else:
                            name_i = line[i][0].upper() + line[i][1:-1].replace('A', 'Z').replace('B', 'Y')
                    elif 'cycleganc' in args.SourceDir:
                        name_i = line[i][0].upper() + line[i][1:-1].replace('A', 'X').replace('B', 'Y').replace('C', 'Z')
                    else:
                        if line[i][0].upper == 'D' or line[i][0].upper == 'G':
                            name_i = line[i][0].upper() + line[i][1:-1].replace('A', 'Y').replace('B', 'X').replace('C', 'Z')
                        else:
                            name_i = line[i][0].upper() + line[i][1:-1].replace('A', 'X').replace('B', 'Y').replace('C', 'Z')
                    idx_name.append(name_i)
                    losses[name_i] = []
                    i += 2
                flag = 0
            
            for i in range(len(idx_name)):
                losses[idx_name[i]].append(float(line[2*i+9]))

valid_idx = [] # idx of losses will be printed out.
if args.CycleLossOnly:
    for i in range(len(idx_name)):
        if "Cycle" in idx_name[i]:
            valid_idx.append(i)
else:
    for i in range(len(idx_name)):
        valid_idx.append(i)

max_loss_value = 1.
if args.Normalized:
    for i in range(len(valid_idx)):
        if max(losses[idx_name[valid_idx[i]]])> max_loss_value:
            max_loss_value = max(losses[idx_name[valid_idx[i]]])

plt.figure()
for i in range(len(valid_idx)):
        if i > 9:
            plt.plot(np.array(losses[idx_name[valid_idx[i]]])/float(max_loss_value), '--', label=idx_name[valid_idx[i]])
        else:
            plt.plot(np.array(losses[idx_name[valid_idx[i]]])/float(max_loss_value), label=idx_name[valid_idx[i]])
# plt.plot(d_loss, 'ro', label='d_loss')
# plt.plot(g_loss, 'go', label='g_loss')
plt.legend(ncol=2)
# plt.legend(loc='upper right')
# plt.legend(bbox_to_anchor=(plt.xlim()[1], plt.ylim()[1]))
# plt.title(args.SourceDir)
plt.ylim(0.0, 1.1)
plt.xlim(0, 100)
plt.savefig('./loss_plot/' + args.SourceDir + '.png')
# plt.show()
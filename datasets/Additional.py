# authors : Jinglan Liu
# license : MIT
from __future__ import print_function

import os, shutil
import numpy as np
import matplotlib.pyplot as plt


def check_format(filepath):
    files = os.listdir(filepath)
    for fi in files:
        fi_list = fi.split('_')
        if len(fi_list) != 5:
            print(fi)
        if fi_list[1] != 'DICOM':
            print(fi)

def check_pair(filepath1, filepath2):
    files1 = sorted(os.listdir(filepath1))
    files2 = sorted(os.listdir(filepath2))
    if len(files1) != len(files2):
        print(filepath1+':'+len(files1))
        print(filepath2+':'+len(files2))
    for i in range(len(files1)):
        file_list1 = files1[i].split('_')
        file_list2 = files2[i].split('_')
        for j in range(len(file_list1)):
            if j != 3:
                if file_list1[j] != file_list2[j]:
                    print(files1[i], files2[i])
                    exit(0) 

def prodC(filepath1, filepath2, filepath3):
    files1 = sorted(os.listdir(filepath1))
    files2 = sorted(os.listdir(filepath2))
    for i in range(len(files1)):
        tmp3 = int((float(files1[i].split('_')[3][1:]) + float(files2[i].split('_')[3][1:]))/2)
        tmp_name_list = files1[i].strip().split('_')
        tmp_name_list[3] = 'S'+str(tmp3)
        files3 = '_'.join(tmp_name_list)
        # files3 = files1[i].split('_')[0]+'_'+files1[i].split('_')[1]+files1[i].split('_')[2]++files1[i].split('_')[4]
        num1 = np.load(os.path.join(filepath1, files1[i])).astype(np.float32)
        num2 = np.load(os.path.join(filepath2, files2[i])).astype(np.float32)
        num3 = (num1+num2)/2.
        np.save(os.path.join(filepath3, files3), num3)

def prodEven(filepath1, filepath2, filepath3, alpha):
    files1 = sorted(os.listdir(filepath1))
    files2 = sorted(os.listdir(filepath2))
    mean_list = []
    std_list = []
    for i in range(len(files1)):
        tmp3 = int(float(files1[i].split('_')[3][1:])*alpha + float(files2[i].split('_')[3][1:])*(1-alpha))
        tmp_name_list = files1[i].strip().split('_')
        tmp_name_list[3] = 'S'+str(tmp3)
        files3 = '_'.join(tmp_name_list)
        num1 = np.load(os.path.join(filepath1, files1[i])).astype(np.float32)
        num2 = np.load(os.path.join(filepath2, files2[i])).astype(np.float32)
        num3 = alpha*num1+(1-alpha)*num2
        np.save(os.path.join(filepath3, files3), num3)

def AnalyA(filepath1, filepath2, alpha):#, filepath3
    files1 = sorted(os.listdir(filepath1))
    files2 = sorted(os.listdir(filepath2))
    mean_list = []
    std_list = []
    for i in range(len(files1)):
        tmp3 = int((float(files1[i].split('_')[3][1:]) + float(files2[i].split('_')[3][1:]))/2)
        files3 = files1[i].split('_')[0]+'_'+files1[i].split('_')[1]+'_'+files1[i].split('_')[2]+'_S'+str(tmp3)+'_'+files1[i].split('_')[4]
        num1 = np.load(os.path.join(filepath1, files1[i])).astype(np.float32)
        num2 = np.load(os.path.join(filepath2, files2[i])).astype(np.float32)
        num3 = alpha*num1+(1-alpha)*num2
        # np.save(os.path.join(filepath3, files3), num3)
        mean_list.append(np.mean(num3))
        std_list.append(np.std(num3))
    mean_np = np.array(mean_list).reshape(-1)
    std_np = np.array(std_list).reshape(-1)
    mean_std = np.vstack((mean_np, std_np))
    # np.save(os.path.join(os.path.join(filepath3, str(alpha)+'.npy')), np.array(mean_std))
    return mean_std

def partial_copy(sfilepath, tfilepath, limit):
    files = sorted(os.listdir(sfilepath))
    for i in range(limit):
        shutil.copyfile(os.path.join(sfilepath, files[i]), os.path.join(tfilepath, files[i]))
        
# check_format('./tw_np_all/trainA')
# check_format('./tw_np_all/trainB')

# check_pair('./tw_np_all/trainA', './tw_np_all/trainB')

# prodC('./tw_np_all/trainA', './tw_np_all/trainB', './tw_np_all/trainC')
# prodC('./tw_np_all/trainC', './tw_np_all/trainB', './tw_np_all/trainD')

# alpha_base = 0.2
# NumTested = int(1/alpha_base)+1
# mean_std_rec = []
# for i in range(NumTested):
#     print(i, ':', alpha_base*i)
#     mean_std_rec.append(AnalyA('./tw_np_all/trainA', './tw_np_all/trainB', alpha_base*i)) #'./tw_np_all/merge', 

# n_bins = 100
# rows = mean_std_rec[0].shape[0]
# fig, axs = plt.subplots(rows, NumTested, sharey=True, tight_layout=True)

# for i in range(NumTested):
#     for j in range(rows):
#         axs[j, i].hist(mean_std_rec[i][j,:], bins = n_bins)
# plt.title('1s row: mean. 2nd row: std')
# plt.show()

# limit = 200
# partial_copy('./tw_np_all/trainA', './tw_np_'+str(limit)+'/trainA', limit)
# partial_copy('./tw_np_all/trainB', './tw_np_'+str(limit)+'/trainB', limit)
# partial_copy('./tw_np_all/trainC', './tw_np_'+str(limit)+'/trainC', limit)
# partial_copy('./tw_np_all/trainD', './tw_np_'+str(limit)+'/trainD', limit)

prodEven('./tw_np_even_200/trainA', './tw_np_even_200/trainB', './tw_np_even_200/trainC', 0.33)
prodEven('./tw_np_even_200/trainA', './tw_np_even_200/trainB', './tw_np_even_200/trainD', 0.66)

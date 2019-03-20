import os
import numpy as np
import matplotlib.pyplot as plt

def image_119_rec(image_numpy, image_path):
    # print('A')
    plt.clf()
    plt.imshow(np.squeeze(image_numpy), cmap=plt.cm.bone)
    currentAxisA = plt.gca()
    # rectA0 = patches.Rectangle((15, 50), 25, 40, linewidth=1, edgecolor='r', facecolor='none')
    rectA1 = patches.Rectangle((18, 112), 18, 20, linewidth=1, edgecolor='r', facecolor='none')
    rectA2 = patches.Rectangle((50, 230), 30, 20, linewidth=1, edgecolor='r', facecolor='none')
    # currentAxisA.add_patch(rectA0)
    currentAxisA.add_patch(rectA1)
    currentAxisA.add_patch(rectA2)
    # plt.axis('off')
    currentAxisA.axes.get_xaxis().set_visible(False)
    currentAxisA.axes.get_yaxis().set_visible(False)
    currentAxisA.spines['left'].set_color('none')
    currentAxisA.spines['bottom'].set_color('none')
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0.0)
    plt.show()

    # mean_str = [str(round(np.mean(image_numpy[50:50+40, 15:15+25]),2)), str(round(np.mean(image_numpy[112:112+20, 18:18+18]),2)), str(round(np.mean(image_numpy[230:230+20, 50:50+30]),2))]
    # std_str = [str(round(np.std(image_numpy[50:50+40, 15:15+25]),2)), str(round(np.std(image_numpy[112:112+20, 18:18+18]),2)), str(round(np.std(image_numpy[230:230+20, 50:50+30]),2))]
    mean_str = [str(round(np.mean(image_numpy[112:112+20, 18:18+18]),2)), str(round(np.mean(image_numpy[230:230+20, 50:50+30]),2))]
    std_str = [str(round(np.std(image_numpy[112:112+20, 18:18+18]),2)), str(round(np.std(image_numpy[230:230+20, 50:50+30]),2))]

    return mean_str, std_str

image_names = ["199_fbp_atf_real_A.npy", "199_fbp_atf_fake_B_A.npy", "200_fbp_atf_real_A.npy", "200_fbp_atf_fake_B_A.npy", "201_fbp_atf_real_A.npy", "201_fbp_atf_fake_B_A.npy", "506_fbp_atf_real_A.npy", "506_fbp_atf_fake_B_A.npy"]

npy_dir = "../results"
image_dir = "../results/miccai"

experiment_name = "twnp200c_cyclegan4c_batch2"

for image in image_names:
    npy_path = os.path.join(npy_dir, experiment_name, "test_latest/images", image)
    image_path = os.path.join(image_dir, experiment_name, "images", image)

    [mean_str, std_str] = image_119_rec(np.load(npy_path), image_path)
    print("mean: " + ",".join(mean_str))
    print("std: " + ",",join(std_str))

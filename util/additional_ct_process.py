import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

edge_color = 'r'
line_width = 2

def image_119_rec(image_numpy, image_path, orig_mean):
    # print('A')
    image_npy = image_numpy/(image_numpy.mean()/orig_mean)
    print(image_npy.mean())
    plt.clf()
    plt.imshow(np.squeeze(image_npy), cmap=plt.cm.bone)
    currentAxisA = plt.gca()
    # rectA0 = patches.Rectangle((15, 50), 20, 25, linewidth=1, edgecolor='r', facecolor='none')
    rectA1 = patches.Rectangle((25, 112), 18, 20, linewidth=line_width, edgecolor=edge_color, facecolor='none')
    rectA2 = patches.Rectangle((100, 8), 25, 25, linewidth=line_width, edgecolor=edge_color, facecolor='none')
    # currentAxisA.add_patch(rectA0)
    currentAxisA.add_patch(rectA1)
    currentAxisA.add_patch(rectA2)
    # plt.axis('off')
    currentAxisA.axes.get_xaxis().set_visible(False)
    currentAxisA.axes.get_yaxis().set_visible(False)
    currentAxisA.spines['left'].set_color('none')
    currentAxisA.spines['bottom'].set_color('none')
    # plt.show()
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0.0)

    # mean_str = [str(round(np.mean(image_numpy[50:50+25, 15:15+20]),2)), str(round(np.mean(image_numpy[112:112+20, 25:25+18]),2)), str(round(np.mean(image_numpy[8:8+25, 100:100+25]),2))]
    # std_str = [str(round(np.std(image_numpy[50:50+25, 15:15+20]),2)), str(round(np.std(image_numpy[112:112+20, 25:25+18]),2)), str(round(np.std(image_numpy[8:8+25, 100:100+25]),2))]
    mean_str = [str(round(np.mean(image_npy[112:112+20, 25:25+18]),2)), str(round(np.mean(image_npy[8:8+25, 100:100+25]),2))]
    std_str = [str(round(np.std(image_npy[112:112+20, 25:25+18]),2)), str(round(np.std(image_npy[8:8+25, 100:100+25]),2))]

    return mean_str, std_str

def image_201_rec(image_numpy, image_path, orig_mean):
    # print('A')
    image_npy = image_numpy-(image_numpy.mean()-orig_mean)
    print(image_npy.mean())
    plt.clf()
    plt.imshow(np.squeeze(image_npy), cmap=plt.cm.bone)
    currentAxisA = plt.gca()
    rectA1 = patches.Rectangle((25, 12), 30, 25, linewidth=line_width, edgecolor=edge_color, facecolor='none')
    # rectA2 = patches.Rectangle((138, 173), 20, 15, linewidth=1, edgecolor='r', facecolor='none')
    currentAxisA.add_patch(rectA1)
    # currentAxisA.add_patch(rectA2)
    # plt.axis('off')
    currentAxisA.axes.get_xaxis().set_visible(False)
    currentAxisA.axes.get_yaxis().set_visible(False)
    currentAxisA.spines['left'].set_color('none')
    currentAxisA.spines['bottom'].set_color('none')
    # plt.show()
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0.0)

    mean_str = [str(round(np.mean(image_npy[12:12+25, 25:25+30]),2)), str(round(np.mean(image_npy[173:173+15, 138:138+20]),2))]
    std_str = [str(round(np.std(image_npy[12:12+25, 25:25+30]),2)), str(round(np.std(image_npy[173:173+15, 138:138+20]),2))]

    return mean_str, std_str

def image_506_rec(image_numpy, image_path, orig_mean):
    # print('A')
    image_npy = image_numpy-(image_numpy.mean()-orig_mean)
    print(image_npy.mean())
    plt.clf()
    plt.imshow(np.squeeze(image_npy), cmap=plt.cm.bone)
    currentAxisA = plt.gca()
    rectA1 = patches.Rectangle((20, 142), 25, 20, linewidth=line_width, edgecolor=edge_color, facecolor='none')
    rectA2 = patches.Rectangle((40, 105), 25, 15, linewidth=line_width, edgecolor=edge_color, facecolor='none')
    currentAxisA.add_patch(rectA1)
    currentAxisA.add_patch(rectA2)
    # plt.axis('off')
    currentAxisA.axes.get_xaxis().set_visible(False)
    currentAxisA.axes.get_yaxis().set_visible(False)
    currentAxisA.spines['left'].set_color('none')
    currentAxisA.spines['bottom'].set_color('none')
    # plt.show()
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0.0)

    mean_str = [str(round(np.mean(image_npy[142:142+20, 20:20+25]),2)), str(round(np.mean(image_npy[105:105+15, 40:40+25]),2))]
    std_str = [str(round(np.std(image_npy[142:142+20, 20:20+25]),2)), str(round(np.std(image_npy[105:105+15, 40:40+25]),2))]

    return mean_str, std_str

# if cycle-gan
# image_199_names = ["199_fbp_atf_real_A", "199_fbp_atf_fake_B", "200_fbp_atf_real_A", "200_fbp_atf_fake_B"]
# image_201_names = ["201_fbp_atf_real_A", "201_fbp_atf_fake_B"]
# image_506_names = ["506_fbp_atf_real_A", "506_fbp_atf_fake_B"]

# if multi-step(long distance) or multi-cycle
image_199_names = ["199_fbp_atf_real_A", "199_fbp_atf_fake_B_A", "200_fbp_atf_real_A", "200_fbp_atf_fake_B_A"]
image_201_names = ["201_fbp_atf_real_A", "201_fbp_atf_fake_B_A"]
image_506_names = ["506_fbp_atf_real_A", "506_fbp_atf_fake_B_A"]

# if decoupled, use _fake_B
# image_199_names = ["199_fbp_atf_real_A_fake_B", "199_fbp_atf_fake_B_fake_B", "200_fbp_atf_real_A_fake_B", "200_fbp_atf_fake_B_fake_B"]
# image_201_names = ["201_fbp_atf_real_A_fake_B", "201_fbp_atf_fake_B_fake_B"]
# image_506_names = ["506_fbp_atf_real_A_fake_B", "506_fbp_atf_fake_B_fake_B"]

npy_dir = "."
image_dir = "./miccai"

experiment_name = "twnp200c_cyclegan4c_batch2"
# "twnp200c_cyclegan4c_batch2"
# "decouple_cb200_cyclegan4_iter50_batch2"
# "twnp200_cyclegan_iter50_batch2"
# "twnp200c_cyclegan4cl_batch2"

# Get mean of pixel values of a whole CT image
mean_199 = 0.
mean_200 = 0.
mean_201 = 0.
mean_506 = 0.
for image in image_199_names:
    if ("real_A" in image) and ("199" in image):
        npy_path = os.path.join(npy_dir, experiment_name, "test_latest/images", image + ".npy")
        mean_199 = np.load(npy_path).mean()
for image in image_199_names:
    if ("real_A" in image) and ("200" in image):
        npy_path = os.path.join(npy_dir, experiment_name, "test_latest/images", image + ".npy")
        mean_200 = np.load(npy_path).mean()
for image in image_201_names:
    if ("real_A" in image) and ("201" in image):
        npy_path = os.path.join(npy_dir, experiment_name, "test_latest/images", image + ".npy")
        mean_201 = np.load(npy_path).mean()
for image in image_506_names:
    if ("real_A" in image) and ("506" in image):
        npy_path = os.path.join(npy_dir, experiment_name, "test_latest/images", image + ".npy")
        mean_506 = np.load(npy_path).mean()

print(mean_199)
print(mean_200)
print(mean_201)
print(mean_506)

for image in image_199_names:
    print(image+": ")

    npy_path = os.path.join(npy_dir, experiment_name, "test_latest/images", image + ".npy")
    image_path = os.path.join(image_dir, experiment_name, "images", image + ".png")

    if "199" in image:
        [mean_str, std_str] = image_119_rec(np.load(npy_path), image_path, mean_199)
        print("mean: " + ",".join(mean_str))
        print("std: " + ",".join(std_str))
    elif "200" in image:
        [mean_str, std_str] = image_119_rec(np.load(npy_path), image_path, mean_200)
        print("mean: " + ",".join(mean_str))
        print("std: " + ",".join(std_str))

for image in image_201_names:
    print(image+": ")

    npy_path = os.path.join(npy_dir, experiment_name, "test_latest/images", image + ".npy")
    image_path = os.path.join(image_dir, experiment_name, "images", image + ".png")

    [mean_str, std_str] = image_201_rec(np.load(npy_path), image_path, mean_201)
    print("mean: " + ",".join(mean_str))
    print("std: " + ",".join(std_str))

for image in image_506_names:
    print(image+": ")

    npy_path = os.path.join(npy_dir, experiment_name, "test_latest/images", image + ".npy")
    image_path = os.path.join(image_dir, experiment_name, "images", image + ".png")

    [mean_str, std_str] = image_506_rec(np.load(npy_path), image_path, mean_506)
    print("mean: " + ",".join(mean_str))
    print("std: " + ",".join(std_str))
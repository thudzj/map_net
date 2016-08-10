import numpy as np
from scipy.misc import imread

la = open("labels.txt")
i = 0

data = []
labels = []

for line in la:
    label = int(line.strip())
    labels.append(label)
    i += 1
    f = "imgs/" + str(i) + ".png"
    im = imread(f)
    im = im / np.float32(256)
    data.append(im.reshape((1, im.shape[0], im.shape[1])))

data = np.array(data)
labels = np.array(labels)
np.savez("train", data, labels)

npz_file = np.load("train.npz")
print npz_file['arr_0'].shape
print npz_file['arr_1'].shape
    
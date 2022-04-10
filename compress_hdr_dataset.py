import numpy as np
import glob
from ezexr import imread, imwrite
import scipy.ndimage

# list all files (recursive) in ~/datasets_ssd/LavalIndoor/1942x971/
files = glob.glob("/root/datasets_ssd/LavalIndoor/1942x971/**/*.exr", recursive=True)

for file in files:
    print(file)
    img = imread(file).astype(np.float32)
    ratio = 128 / img.shape[0]
    img = scipy.ndimage.zoom(img, (ratio, ratio, 1), order=0).astype(np.float32)
    imwrite(file, img)
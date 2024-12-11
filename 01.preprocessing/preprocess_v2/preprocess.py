
import re
import math
import argparse
import struct
import gzip
import pathlib
import numpy as np
from PIL import Image, ImageOps

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=pathlib.Path, required=True)
parser.add_argument('--output-train', type=pathlib.Path, required=True)
#parser.add_argument('--output-test', type=pathlib.Path, required=True)
args = parser.parse_args()

image_dir = args.input / 'assets'
label_path = args.input / 'data.csv'

# read all images
image_list = []
for image_path in sorted(image_dir.glob('*.png')):
    with Image.open(image_path) as im:
        gray = im.convert('L')
        image_list.append(gray)

# pad to max then resize to 28x28
W = max(map(lambda im: im.width, image_list))
H = max(map(lambda im: im.height, image_list))
image_list = [
    ImageOps.fit(ImageOps.pad(im, (W, H), color='#fff'), (28, 28))
    for im in image_list
]

# pack images to numpy array and invert color
image_array = 255 - np.stack([np.asarray(im) for im in image_list], axis=0)
assert image_array.dtype == np.uint8, 'dtype should be uint8'

# read label file and join images
labels = []
image_idx = []
with open(label_path, 'r') as fp:
    for l in fp.read().splitlines():
        # check entry format
        m = re.match(r'^,([0-9]+),([0-9])$', l)
        if not m:
            print(f'info: skip invalid label line "{l}"')
            continue
        # check image idx
        idx, label = int(m.group(1)), int(m.group(2))
        if idx < 0 or idx >= len(image_array):
            print(f'info: skip invalid label line "{l}"')
            continue
        image_idx.append(idx)
        labels.append(label)
labels = np.array(labels)

def export_dataset(path, images, labels, n):
    assert len(images) == len(labels), 'images and labels should have same length'
    if len(images) < n:
        # make duplicates to create n samples
        reps = math.ceil(n / len(images))
        images = np.tile(images, (reps, 1, 1))[:n]
        labels = np.tile(labels, (reps,))[:n]
    elif len(images) > n:
        images = images[:n]
        labels = labels[:n]

    # add dummy 7 columns to labels
    labels = (
        np.hstack((labels.reshape(-1, 1), np.zeros((labels.size, 7))))
        .astype(np.int32)
    )

    # write gzip file
    if not path.exists():
        path.mkdir()
    with gzip.open(path / 'images.gz', 'wb') as fp:
        fp.write(struct.pack('>iiii', 2051, *images.shape))
        fp.write(images.tobytes())
    with gzip.open(path / 'labels.gz', 'wb') as fp:
        fp.write(struct.pack('>iii', 3074, *labels.shape))
        fp.write(labels.astype(np.dtype('>i4')).tobytes())

export_dataset(args.output_train, image_array[image_idx], labels, n=60000)
#train_idx = [i for i, ii in enumerate(image_idx) if ii < 90] # 150
#test_idx = [i for i, ii in enumerate(image_idx) if ii >= 90]
#export_dataset(args.output_train, image_array[train_idx], labels[train_idx], n=60000)
#export_dataset(args.output_test, image_array[test_idx], labels[test_idx], n=10000)

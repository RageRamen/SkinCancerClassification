import pandas as pd
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow
import os

metadata = pd.read_csv("Dataset/HAM10000_metadata.csv")
print(metadata.head())

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

folder1 = "Dataset/HAM10000_images_part_1"
folder2 = "Dataset/HAM10000_images_part_2"

image_path_dict = {}
for index, row in metadata.iterrows():
    img_id = row['image_id']

    if os.path.exists(os.path.join(folder1, img_id + '.jpg')):
        image_path_dict[img_id] = os.path.join(folder1, img_id + '.jpg')
    elif os.path.exists(os.path.join(folder2, img_id + '.jpg')):
        image_path_dict[img_id] = os.path.join(folder2, img_id + '.jpg')
    #else:
    #    raise ValueError(f"Image {img_id} not found in either folder")


metadata['path'] = metadata['image_id'].map(image_path_dict)

found_count = metadata['path'].notna().sum()
print(f"Found {found_count} images out of {len(metadata)}")
print(metadata[['image_id', 'path']].head())

img = mpimg.imread(metadata['path'][0])
plt.imshow(img)
plt.show()




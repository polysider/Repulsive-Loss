import numpy as np  # For array processing
from os import listdir  # For listing all files in directory
from os.path import expanduser  # For finding home directory
import os
import shutil  # To move images
from PIL import Image  # For Image processing
import cv2 # OpenCV for image resising
from scipy.misc import imsave  # To save images as png
from random import randint

import torch
import pandas as pd
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class SOP(Dataset):
    """
    torch.Dataset interface implementation for the Stanford Online Products dataset
    """
    IMAGE_WIDTH = 256
    IMAGE_HEIGHT = 256

    def __init__(self, root, train=True, transform=None):

        self.train = train
        self.root = os.path.expanduser(root)
        self.transform = transform

        split = 'train' if train else 'test'

        description_file = os.path.join(self.root, 'Ebay_{}.txt'.format(split))

        dataset_df = pd.read_csv(description_file, sep=" ", header=0, names=['image_id', 'class_id', 'super_class_id', 'path'])

        assert dataset_df['path'].apply(lambda x: os.path.isfile(os.path.join(self.root, x))).all(), \
            "Some images referenced in the CSV file were not found"

        image_paths = dataset_df['path'].values
        labels = dataset_df['class_id'].values

        if self.train:
            self.train_data = image_paths
            self.train_labels = labels

        else:
            self.test_data = image_paths
            self.test_labels = labels


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        if self.train:
            img = Image.open(os.path.join(self.root, self.train_data[index]))
            label = self.train_labels[index]
        else:
            img = Image.open(os.path.join(self.root, self.test_data[index]))
            label = self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img)

        img = img.convert('RGB')
        img_array = np.array(img)

        # img = self._process_img(img)
        img_array = cv2.resize(img_array, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT))
        img = Image.fromarray(img_array)

        if self.transform is not None:
            img = self.transform(img)

        return img, label


    def __len__(self):

        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


    def _process_img(self, img):

        (width, height) = img.size
        print("Image original size is {}".format(img.size))
        if not width == self.IMAGE_WIDTH:
            img = img.crop(((width - self.IMAGE_WIDTH) / 2, 0, width - (width - self.IMAGE_WIDTH) // 2, height))

        if not height == self.IMAGE_HEIGHT:
            img = img.crop(
                (0, (height - self.IMAGE_HEIGHT) / 2, self.IMAGE_WIDTH, height - (height - self.IMAGE_HEIGHT) // 2))
        print("Image resized to {}".format(img.size))
        return img
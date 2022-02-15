# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import os

import albumentations
import cv2
import numpy as np
import pandas as pd
import sklearn
from tensorflow import keras


# -

class Generator(keras.utils.Sequence):
    def __init__(
        self,
        df,
        images_dir,
        batch_size,
        img_size,
        n_chanels,
        cache_dir=None,
        shuffle=False,
        augment=True,
        normalize=False,
    ):
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.img_size = img_size
        self.augment = augment
        self.normalize = normalize
        self.cache_dir = cache_dir
        self.n_chanels = n_chanels
        if self.shuffle:
            self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self):
        """
        Length in batches
        """
        return self.df.shape[0] // self.batch_size

    def __getitem__(self, b_ix):

        X, Y = [], []

        for i in range(self.batch_size):
            x_, y_ = self.get_one(
                i + self.batch_size * b_ix,
            )
            X.append(x_)
            Y.append(y_)
        X = np.array(X)
        Y = np.array(Y)

        return X, Y

    def get_image(self, img_name):
        try:
            img = np.load(self.cache_dir + img_name)
        except:
            img = cv2.imread("/app/_data/train_images/" + img_name)
            img = cv2.resize(
                img, [self.img_size, self.img_size], interpolation=cv2.INTER_LINEAR
            )
            if self.normalize:
                img = (img - img.min()) / (img.max() - img.min())
            if self.n_chanels > 3:
                chanels = np.random.randint(0, 3, self.n_chanels - 3)
                img = np.concatenate(
                    [img, np.stack([img[:, :, i] for i in chanels], axis=-1)], axis=-1
                )
            if self.cache_dir is not None:
                np.save(self.cache_dir + img_name[:-4], img)
        return img

    def get_one(self, ix):
        """
        Get single item by absolute index
        """

        # img
        img_name = self.df.loc[ix, "image"]
        img = self.get_image(img_name)

        # mask
        mask = np.zeros([self.img_size, self.img_size], dtype="uint8")
        x_min = np.int(self.df.loc[ix, "x_min"] * self.img_size)
        y_min = np.int(self.df.loc[ix, "y_min"] * self.img_size)
        x_max = np.int(self.df.loc[ix, "x_max"] * self.img_size) + 1
        y_max = np.int(self.df.loc[ix, "y_max"] * self.img_size) + 1
        mask[y_min:y_max, x_min:x_max] = 1

        # augment
        if self.augment:
            img, mask = self._augment_image(img, mask)
        return img, mask

    def _augment_image(self, image, mask):
        transform = albumentations.Compose(
            [
                albumentations.OneOf(
                    [
                        albumentations.HorizontalFlip(),
                        albumentations.VerticalFlip(),
                        albumentations.augmentations.geometric.rotate.RandomRotate90(),
                    ],
                    p=0.1,
                ),
            ]
        )
        transformed = transform(image=image, mask=mask)
        transformed_image = transformed["image"]
        transformed_mask = transformed["mask"]
        return transformed_image, transformed_mask

    def on_epoch_end(self):
        if self.shuffle:
            return self.df.sample(frac=1, random_state=42).reset_index(drop=True)



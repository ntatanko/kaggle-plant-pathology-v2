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

import numpy as np
import pandas as pd

# import warnings
# warnings.filterwarnings("ignore")
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# %matplotlib inline
import sklearn
from sklearn import metrics
import tensorflow as tf
import tensorflow_addons as tfa
from PIL import Image
from tensorflow import keras
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.layers import (
    AveragePooling2D,
    AvgPool2D,
    Conv2D,
    Dense,
    Flatten,
    GlobalAveragePooling2D,
    MaxPooling2D,
)
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
feature_columns = ['complex', 'frog_eye_leaf_spot', 'healthy', 'powdery_mildew', 'rust', 'scab']
# -
feature_columns = ['complex', 'frog_eye_leaf_spot', 'healthy', 'powdery_mildew', 'rust', 'scab']



def get_prediction(predict_train =False, predict_valid=False, all_img = False):
    if all_img:
        all_img = ImageDataGenerator(rescale=1.0 / 255).flow_from_dataframe(
        dataframe=labels,
        directory=IMG_PATH,
        x_col="image",
        y_col=feature_columns,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="raw",
        seed=SEED,
        interpolation = 'bicubic',
        shuffle=False,
    )
        prediction_all = model.predict(all_img)
        prediction = pd.DataFrame(prediction_all, columns=feature_columns).join(pd.DataFrame(all_img._targets, columns=feature_columns), rsuffix='_true', lsuffix = '_pred')
        prediction.index = all_img.filenames
    elif predict_valid:       
        prediction_valid = model.predict(valid)
        prediction = pd.DataFrame(prediction_valid, columns=feature_columns).join(pd.DataFrame(valid._targets, columns=feature_columns), rsuffix='_true', lsuffix = '_pred')
        prediction.index = valid.filenames
    elif predict_train:
        train = train_datagen.flow_from_dataframe(
            dataframe=labels,
            directory=IMG_PATH,
            x_col="image",
            y_col=feature_columns,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="raw",
            subset="training",
            seed=SEED,
            interpolation = 'bicubic',
            shuffle=False
        )
        prediction_train = model.predict(train)
        prediction = pd.DataFrame(prediction_train, columns=feature_columns).join(pd.DataFrame(train._targets, columns=feature_columns), rsuffix='_true', lsuffix = '_pred')
        prediction.index = train.filenames
    return prediction


def get_metrics(prediction_df, show=True):
    metrics_df = pd.DataFrame()
    for col in feature_columns:
        pres = metrics.precision_score(prediction_df[col+'_true'], prediction_df[col+'_pred']>0.5)
        rec = metrics.recall_score(prediction_df[col+'_true'], prediction_df[col+'_pred']>0.5)
        f1 = metrics.f1_score(prediction_df[col+'_true'], prediction_df[col+'_pred']>0.5)
        metrics_df.loc['precision', col] = pres
        metrics_df.loc['recall', col] = rec
        metrics_df.loc['f1_score', col] = f1
    if show:
        print(metrics_df)
    return metrics_df


def wrong_prediction(prediction_df, treshold = 0.5):
    list_wrong_prediction = []
    prediction = prediction_df.copy()
    for col in feature_columns:
        prediction[col+'_pred'] = prediction[col+'_pred'] > treshold
        prediction[col+'_pred'] = prediction[col+'_pred'].replace({True: 1, False: 0})
        ids = prediction[prediction[col+'_true'] != prediction[col+'_pred']].index.tolist()
        list_wrong_prediction.extend(ids)
    list_wrong_prediction = set(list_wrong_prediction)
    wrong_prediction = prediction.loc[list_wrong_prediction]
    for img in wrong_prediction.index.tolist():
        pred = wrong_prediction.loc[img][:6].values
        real = wrong_prediction.loc[img][6:12].values
        pred_names = ' '.join(list(map(lambda x,y: x*y, pred,feature_columns)))
        pred_names = ' '.join(pred_names.split())
        real_names = ' '.join(list(map(lambda x,y: x*y, real,feature_columns)))
        real_names = ' '.join(real_names.split())
        wrong_prediction.loc[img, 'pred_labels'] = pred_names
        wrong_prediction.loc[img, 'real_labels'] = real_names
    return prediction, wrong_prediction


def plot_wrongs(df, num_im = 20):
    plt.figure(figsize=(15,(num_im//4)*5))
    for i, img in enumerate(df.sample(num_im).index.tolist()):
        image = Image.open(IMG_PATH +img)
        plt.subplot(num_im//4, 4, i+1)
        plt.title('predicted: '+df.loc[img,'pred_labels']+ '\nreal: '+df.loc[img,'real_labels'])
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.show();



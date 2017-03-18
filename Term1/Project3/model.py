import os
import csv
import cv2
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers import Cropping2D, Lambda, ELU
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import numpy as np
import tensorflow as tf
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle

def data_gen(samples, batch_size=32):
    num_samples = len(samples)
    print ("number of samples  = ", num_samples)
    while 1: # Loop forever so the generator never terminates
        #shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = 'Udacity_Data/data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)


def get_data():
    """
    capture image and steering data
    """
    samples = []
    with open('Udacity_Data/data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

    return samples

def model_comma_ai(camera_format, crop=None):
    """
    model developed by comma.ai: https://github.com/commaai/research
    """
    row, col, ch = camera_format
    if crop is None:
        crop_top, crop_bot, crop_left, crop_right = 0
    else:
        crop_top, crop_bot, crop_left, crop_right = crop
    row_cropped = row - crop_top - crop_bot
    col_cropped = col - crop_left - crop_right

    model = Sequential()
    model.add(Cropping2D(cropping = ((crop_top, crop_bot), (crop_left, crop_right)), input_shape = (row, col, ch), data_format = "channels_last"))
    model.add(Lambda(lambda x: x/127.5 - 1.,
                        input_shape=(row_cropped, col_cropped, ch),
                        output_shape=(row_cropped, col_cropped, ch)))

    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())

    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())

    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())

    model.add(Dropout(1))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(1))
    model.add(ELU())
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    return model

def nvidia_model(camera_format, crop=None):
    """
    model developed by nvidia:
    https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    """

    pass

def train_model(model, data):
    """
    train model on supplied data
    """
    train_samples, validation_samples = train_test_split(data, test_size=0.2)
    train_generator = data_gen(train_samples, batch_size=32)
    validation_generator = data_gen(validation_samples, batch_size=32)

    model.fit_generator(train_generator,
                        samples_per_epoch=len(train_samples),
                        validation_data=validation_generator,
                        nb_val_samples=len(validation_samples),
                        nb_epoch=5)

    return model

if __name__ == "__main__":
    data = get_data()
    model = model_comma_ai((160, 320, 3), crop=(50, 20, 0, 0))
    train_model(model, data)
    model.save('comma_ai_model.h5')
    # predict_on_new_data()

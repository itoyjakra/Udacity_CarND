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
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def data_gen(samples, batch_size=32):
    num_samples = len(samples)
    print ("number of samples  = ", num_samples)
    while 1:
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

def generate_training_batch(images, angles, batch_size):
    """
    yield a batch of data for training
    """
    while 1:
        batch_images = []
        batch_angles = []
        for i in range(batch_size):
            index = np.random.randint(len(angles))
            batch_angles.append(angles[index])
            img = cv2.imread(images[index])
            batch_images.append(img)

        yield (np.array(batch_images), np.array(batch_angles))

def get_log_data(steering_offset=0.3, include_center=True, dir_name='Udacity_Data/data', log_file='driving_log.csv'):
    """
    collect image file names and steering angles
    """
    images = []
    angles = []
    with open(dir_name+'/'+log_file) as csvfile:
        for center_img, left_img, right_img, steering_angle, _, _, speed in csv.reader(csvfile):
            center_img = dir_name + '/' + center_img.strip()
            left_img = dir_name + '/' + left_img.strip()
            right_img = dir_name + '/' + right_img.strip()
            if include_center:
                images.extend([center_img, left_img, right_img])
                angles.extend([float(steering_angle), float(steering_angle)+steering_offset, float(steering_angle)-steering_offset])
            else:
                images.extend([left_img, right_img])
                angles.extend([steering_angle+steering_offset, steering_angle-steering_offset])

    return (images, angles)

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

    model.add(Convolution2D(16, (8, 8), strides=(4, 4), padding="same"))
    model.add(ELU())

    model.add(Convolution2D(32, (5, 5), strides=(2, 2), padding="same"))
    model.add(ELU())

    model.add(Convolution2D(64, (5, 5), strides=(2, 2), padding="same"))
    model.add(Flatten())

    model.add(Dropout(0.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(ELU())
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    return model

def model_nvidia(camera_format, crop=None):
    """
    model developed by nvidia:
    https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    """
    row, col, ch = camera_format
    if crop is None:
        crop_top = crop_bot = crop_left = crop_right = 0
    else:
        crop_top, crop_bot, crop_left, crop_right = crop
    row_cropped = row - crop_top - crop_bot
    col_cropped = col - crop_left - crop_right

    model = Sequential()
    model.add(Cropping2D(cropping = ((crop_top, crop_bot), (crop_left, crop_right)), input_shape = (row, col, ch), data_format = "channels_last"))

    model.add(Lambda(lambda x: x/127.5 - 1.,
                        input_shape=(row_cropped, col_cropped, ch),
                        output_shape=(row_cropped, col_cropped, ch)))

    model.add(Convolution2D(24, (5, 5), strides=(2, 2), padding="valid"))
    model.add(ELU())

    model.add(Convolution2D(36, (5, 5), strides=(2, 2), padding="valid"))
    model.add(ELU())

    model.add(Convolution2D(48, (5, 5), strides=(2, 2), padding="valid"))
    model.add(ELU())

    model.add(Convolution2D(64, (3, 3), strides=(1, 1), padding="valid"))
    model.add(ELU())

    model.add(Convolution2D(64, (3, 3), strides=(1, 1), padding="valid"))
    model.add(ELU())
    model.add(Flatten())

    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(ELU())

    model.add(Dense(50))
    model.add(Dropout(0.5))
    model.add(ELU())

    model.add(Dense(10))
    model.add(Dropout(0.5))
    model.add(ELU())

    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    return model

def train_model(model, data):
    """
    train model on supplied data
    """
    X, y = data
    train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2)
    #train_samples, validation_samples = train_test_split(data, test_size=0.2)
    #train_generator = data_gen(train_samples, batch_size=32)
    #validation_generator = data_gen(validation_samples, batch_size=32)
    #train_generator = generate_training_batch(train_X, train_y, batch_size=32)
    #validation_generator = generate_training_batch(val_X, val_y, batch_size=32)
    train_generator = generate_training_batch(X, y, batch_size=32)

    model.fit_generator(train_generator, samples_per_epoch=8000, nb_epoch=2)
    '''
                        steps_per_epoch=len(X),
                        #steps_per_epoch=len(train_X),
                        #validation_data=validation_generator,
                        #validation_steps=len(val_X),
                        epochs=2)
    '''

    return model

if __name__ == "__main__":
    #data = get_data()
    data = get_log_data(steering_offset=0.3, include_center=True, dir_name='Udacity_Data/data', log_file='driving_log.csv')
    model = model_nvidia((160, 320, 3), crop=(50, 20, 0, 0))
    #model = model_comma_ai((160, 320, 3), crop=(50, 20, 0, 0))
    train_model(model, data)
    model.save('nvidia_model_03.h5')
    # predict_on_new_data()

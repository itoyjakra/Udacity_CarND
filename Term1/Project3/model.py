from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import numpy as np
import tensorflow as tf

def data_gen(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


def get_data():
    """
    capture image and steering data
    """
    pass

def build_model(camera_format):
    """
    create a regression deep network model
    """
    ch, row, col = camera_format # 3, 160, 320  # camera format

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
                        input_shape=(ch, row, col),
                        output_shape=(ch, row, col)))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    return model

def train_model(model):
    """
    train model on training data
    """
    train_data_gen = data_gen(train_samples, batch_size=32)
    validation_generator = data_gen(validation_samples, batch_size=32)

    model.fit_generator(train_generator,
                        samples_per_epoch=len(train_samples),
                        validation_data=validation_generator,
                        nb_val_samples=len(validation_samples),
                        nb_epoch=3)

if __name__ == "__main__":
    get_data()
    model = build_model((3, 160, 320))
    train_model()
    predict_on_new_data()

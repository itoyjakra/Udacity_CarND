# Load pickled data
import pickle

training_file = 'train_p2.p'
testing_file = 'test_p2.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

n_train = len(train['features'])
n_test = len(test['features'])
image_shape = train['features'].shape[1:]
n_classes = len(set(train['labels']))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

from random import random
import numpy as np
import tensorflow as tf

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def convertToOneHot(vector, num_classes=None):
    """
    Converts an input 1-D vector of integers into an output
    2-D array of one-hot vectors, where an i'th input value
    of j will set a '1' in the i'th row, j'th column of the
    output array.

    Example:
        v = np.array((1, 0, 4))
        one_hot_v = convertToOneHot(v)
        print one_hot_v

        [[0 1 0 0 0]
         [1 0 0 0 0]
         [0 0 0 0 1]]
    """

    assert isinstance(vector, np.ndarray)
    assert len(vector) > 0

    if num_classes is None:
        num_classes = np.max(vector)+1
    else:
        assert num_classes > 0
        assert num_classes >= np.max(vector)

    result = np.zeros(shape=(len(vector), num_classes))
    result[np.arange(len(vector)), vector] = 1
    return result.astype(int)


def conv2d(x, W, b, stride=1, padding='VALID'):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, padding=padding, strides=[1, stride, stride, 1])
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2, stride=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, stride, stride, 1], padding='SAME')


def min_max_scale(image_data, low=0.1, high=0.9):
    return low + (image_data - np.min(image_data)) * (high - low) / (np.max(image_data) - np.min(image_data))

def inception(input_layer):
    # input 13x13x32
    conv_1x1 = conv2d(input_layer, weights['winc1'], biases['binc1'], padding='SAME') # 13x13x16
    conv_3x3 = conv2d(conv_1x1, weights['winc3'], biases['binc3'], padding='SAME') # 13x13x32
    conv_5x5 = conv2d(conv_1x1, weights['winc5'], biases['binc5'], padding='SAME') # 13x13x32
    max_pool = maxpool2d(input_layer, k=2, stride=1) # 13x13x32
    max_pool_conv_1x1 = conv2d(max_pool, weights['winpc1'], biases['binpc1'], padding='SAME') # 13x13x32
    conv_d1x1 = conv2d(input_layer, weights['windc1'], biases['bindc1'], padding='SAME') # 13x13x16
    output_layer = tf.concat(3, [conv_3x3, conv_5x5, conv_d1x1, max_pool_conv_1x1]) # 13x13x112
    return output_layer

def Network(x, dropout):
    
    x = tf.reshape(x, (-1, image_shape[0], image_shape[1], 1))
    
    # Conv Layer 1
    conv1 = conv2d(x, weights['wc1'], biases['bc1']) # 26x26x32
    conv1 = maxpool2d(conv1, k=2) # 13x13x32

    '''
    # Conv Layer 2
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)
    #conv2 = tf.nn.dropout(conv2, dropout)
    '''

    # Inception Layer
    incp1 = inception(conv1)

    # FC Layer 1
    fc1 = tf.contrib.layers.flatten(incp1)
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)

    # FC Layer 2
    fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, dropout)

    # Out Layer
    out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
    
    return out

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy, total_loss = 0, 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        loss, accuracy =  sess.run([loss_op, accuracy_op], feed_dict={X: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * batch_x.shape[0])
        total_loss     += (loss * batch_x.shape[0])
    return total_loss / num_examples, total_accuracy / num_examples

def find_error(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy, total_loss = 0, 0
    sess = tf.get_default_session()
    wrong_list = np.array([])
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        wrong =  sess.run(wrong_prediction, feed_dict={X: batch_x, y: batch_y, keep_prob: 1.0})
        #print (offset, len(batch_y[wrong].argmax(axis=1)))
        wrong_list = np.append(wrong_list, batch_y[wrong].argmax(axis=1))
    return wrong_list


# convert images to grayscale
features_train = rgb2gray(train['features'])
print(train['features'].shape, features_train.shape)
labels_train = train['labels']
print (labels_train.shape)

features_test = rgb2gray(test['features'])
print(test['features'].shape, features_test.shape)
labels_test = test['labels']
print (labels_test.shape)

# generate more training data by transforming the existing images
from scipy import ndimage

'''
'''
new_images_1 = np.array([ndimage.rotate(image, -10, reshape=False, mode='nearest') for image in features_train])
new_images_2 = np.array([ndimage.rotate(image, 20, reshape=False, mode='nearest') for image in features_train])
new_images_3 = np.array([ndimage.shift(image, (5, 5), mode='nearest') for image in features_train])
new_images_4 = np.array([ndimage.shift(image, (-5, -5), mode='nearest') for image in features_train])

features_train = np.vstack((features_train, new_images_1, new_images_2, new_images_3, new_images_4))
labels_train = np.array(list(labels_train)*5)
'''
'''
print ('new training set = ', features_train.shape)
print ('new label set = ', labels_train.shape)

# flatten the images
features_train = features_train.reshape(len(features_train), -1)
features_test = features_test.reshape(len(features_test), -1)

# scale the images
features_train = min_max_scale(features_train)
features_test = min_max_scale(features_test)

# split train set into training and validation
from sklearn.cross_validation import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(features_train, labels_train, test_size=0.3, random_state=42)
print (X_train.shape, y_train.shape, X_valid.shape, y_valid.shape)

y_train = convertToOneHot(y_train)
y_valid = convertToOneHot(y_valid)
labels_test = convertToOneHot(labels_test)

# weights and biases
f = 7
weights = {
    'wc1': tf.get_variable('wc1', shape=([f, f, 1, 32]), initializer=tf.contrib.layers.xavier_initializer_conv2d()),
    'winc1': tf.get_variable('winc1', shape=([1, 1, 32, 16]), initializer=tf.contrib.layers.xavier_initializer_conv2d()),
    'winc3': tf.get_variable('winc3', shape=([3, 3, 16, 32]), initializer=tf.contrib.layers.xavier_initializer_conv2d()),
    'winc5': tf.get_variable('winc5', shape=([5, 5, 16, 32]), initializer=tf.contrib.layers.xavier_initializer_conv2d()),
    'winpc1': tf.get_variable('winpc1', shape=([1, 1, 32, 32]), initializer=tf.contrib.layers.xavier_initializer_conv2d()),
    'windc1': tf.get_variable('widnc1', shape=([1, 1, 32, 16]), initializer=tf.contrib.layers.xavier_initializer_conv2d()),
    'wd1': tf.get_variable('wd1', shape=([13*13*112, 1024]), initializer=tf.contrib.layers.xavier_initializer_conv2d()),
    'wd2': tf.get_variable('wd2', shape=([1024, 256]), initializer=tf.contrib.layers.xavier_initializer_conv2d()),
    'out': tf.get_variable('w_out', shape=([256, n_classes]), initializer=tf.contrib.layers.xavier_initializer_conv2d())
}

biases = {
    'bc1': tf.get_variable('bc1', shape=([32]), initializer=tf.contrib.layers.xavier_initializer()),
    'binc1': tf.get_variable('binc1', shape=([16]), initializer=tf.contrib.layers.xavier_initializer_conv2d()),
    'binc3': tf.get_variable('binc3', shape=([32]), initializer=tf.contrib.layers.xavier_initializer_conv2d()),
    'binc5': tf.get_variable('binc5', shape=([32]), initializer=tf.contrib.layers.xavier_initializer_conv2d()),
    'binpc1': tf.get_variable('binpc1', shape=([32]), initializer=tf.contrib.layers.xavier_initializer_conv2d()),
    'bindc1': tf.get_variable('bindc1', shape=([16]), initializer=tf.contrib.layers.xavier_initializer_conv2d()),
    'bd1': tf.get_variable('bd1', shape=([1024]), initializer=tf.contrib.layers.xavier_initializer()),
    'bd2': tf.get_variable('bd2', shape=([256]), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('b_out', shape=([n_classes]), initializer=tf.contrib.layers.xavier_initializer())
}


# Create the Graph

#keep_prob = 0.80
X = tf.placeholder(tf.float32, (None, image_shape[0]*image_shape[1]))
y = tf.placeholder(tf.float32, (None, n_classes))
keep_prob = tf.placeholder(tf.float32)

fc2 = Network(X, keep_prob)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(fc2, y))
opt = tf.train.AdamOptimizer(learning_rate=1.0e-3)
train_op = opt.minimize(loss_op)
correct_prediction = tf.equal(tf.argmax(fc2, 1), tf.argmax(y, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
wrong_prediction = tf.not_equal(tf.argmax(fc2, 1), tf.argmax(y, 1))


# Feed dicts for training, validation, and test session
train_feed_dict = {X: X_train, y: y_train}
valid_feed_dict = {X: X_valid, y: y_valid}


EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 0.0005


from time import time


# Measurements use for graphing loss and accuracy

EPOCHS = 50
dropout_keep_prob = 0.4

train_acc_epoch = []
train_loss_epoch = []
valid_acc_epoch = []
valid_loss_epoch = []

with tf.Session() as sess:
        #sess.run(tf.global_variables_initializer())
        sess.run(tf.initialize_all_variables())
        steps_per_epoch = len(X_train) // BATCH_SIZE
        num_examples = steps_per_epoch * BATCH_SIZE

        # Train model
        train_start_time = time()
        for i in range(EPOCHS):
            start_time = time()
            for step in range(steps_per_epoch):
                #tf.add_check_numerics_ops()
                batch_x = X_train[step*BATCH_SIZE:(step+1)*BATCH_SIZE,...]
                batch_y = y_train[step*BATCH_SIZE:(step+1)*BATCH_SIZE]
                loss = sess.run(train_op, feed_dict={X: batch_x, y: batch_y, keep_prob: dropout_keep_prob})
                
            tra_loss, tra_acc = evaluate(X_train, y_train)
            train_acc_epoch.append(tra_acc)
            train_loss_epoch.append(tra_loss)

            val_loss, val_acc = evaluate(X_valid, y_valid)
            valid_acc_epoch.append(val_acc)
            valid_loss_epoch.append(val_loss)

            print("EPOCH {} ...".format(i+1))
            print("Training loss = {:.3f}   Validation loss = {:.3f}".format(tra_loss, val_loss))
            print("Training accuracy = {:.3f}   Validation accuracy = {:.3f}".format(tra_acc, val_acc))
            print("time taken = {:.1f} s".format(time() - start_time))
            print()
        print("total training time = {:.1f} s".format(time() - train_start_time))

        # Evaluate on the test data
        
        test_loss, test_acc = evaluate(features_test, labels_test)
        wrong_pred = find_error(features_test, labels_test)
        f = np.bincount(wrong_pred.astype(int))
        #print (sorted(zip(range(len(f)), f), key=lambda x: x[1]))
        print ("freq. wrong preds ", np.bincount(wrong_pred.astype(int)))
        print("Test loss = {:.3f}".format(test_loss))
        print("Test accuracy = {:.3f}".format(test_acc))
        
        # dump the trends into a file
        import pandas as pd
        trends = {'training_loss': train_loss_epoch, 'validation_loss': valid_loss_epoch,
                  'training_accuracy': train_acc_epoch, 'validation_accuracy': valid_acc_epoch}
        df_trend = pd.DataFrame(trends)
        df_trend.to_csv("trends.csv", index=False)

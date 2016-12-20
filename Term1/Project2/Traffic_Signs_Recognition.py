
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree
# 
# ## Deep Learning
# 
# ## Project: Build a Traffic Sign Recognition Classifier
# 
# In this notebook, a template is provided for you to implement your functionality in stages which is required to successfully complete this project. If additional code is required that cannot be included in the notebook, be sure that the Python code is successfully imported and included in your submission, if necessary. Sections that begin with **'Implementation'** in the header indicate where you should begin your implementation for your project. Note that some sections of implementation are optional, and will be marked with **'Optional'** in the header.
# 
# In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.
# 
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

# ---
# ## Step 0: Load The Data

# In[1]:

# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = 'train_p2.p'
testing_file = 'test_p2.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']


# ---
# 
# ## Step 1: Dataset Summary & Exploration
# 
# The pickled data is a dictionary with 4 key/value pairs:
# 
# - `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
# - `'labels'` is a 2D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
# - `'sizes'` is a list containing tuples, (width, height) representing the the original width and height the image.
# - `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**
# 
# Complete the basic data summary below.

# In[2]:

### Replace each question mark with the appropriate value.

# TODO: Number of training examples
n_train = len(train['features'])

# TODO: Number of testing examples.
n_test = len(test['features'])

# TODO: What's the shape of an traffic sign image?
image_shape = train['features'].shape[1:]

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(set(train['labels']))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


# Visualize the German Traffic Signs Dataset using the pickled file(s). This is open ended, suggestions include: plotting traffic sign images, plotting the count of each sign, etc.
# 
# The [Matplotlib](http://matplotlib.org/) [examples](http://matplotlib.org/examples/index.html) and [gallery](http://matplotlib.org/gallery.html) pages are a great resource for doing visualizations in Python.
# 
# **NOTE:** It's recommended you start with something simple first. If you wish to do more, come back to it after you've completed the rest of the sections.

# In[3]:

### Data exploration visualization goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
#get_ipython().magic('matplotlib inline')


# In[4]:

from random import random
import numpy as np
import tensorflow as tf


# In[ ]:

np.nonzero(np.bincount(train['labels']))


# In[ ]:

np.bincount(train['labels'])


# In[ ]:

'''
for (label, freq) in zip(range(n_classes), np.bincount(train['labels'])):
    print (label, freq)
'''


# In[ ]:

#plt.bar(range(n_classes), np.bincount(train['labels']))


# In[ ]:

#plt.hist(train['labels'], bins=len(set(train['labels'])))


# In[ ]:

# create a dictionary of labels
from collections import defaultdict
label_dict = {}
label_dict = defaultdict(lambda: [], label_dict)
N = len(train['features'])
for i in range(N):
    label_dict[train['labels'][i]].append(i)


# In[ ]:

# select a few plots randomly from each class and plot
'''
n_cols = 8
plt.figure(figsize=[20, 100])
for i in range(n_classes):
    for j in range(n_cols):
        plt.subplot(n_classes, n_cols, i*n_cols+j+1)
        n = int(random() * len(label_dict[i]))
        image = train['features'][label_dict[i][n]]
        plt.imshow(image)
        plt.axis('off')
'''


# ----
# 
# ## Step 2: Design and Test a Model Architecture
# 
# Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).
# 
# There are various aspects to consider when thinking about this problem:
# 
# - Neural network architecture
# - Play around preprocessing techniques (normalization, rgb to grayscale, etc)
# - Number of examples per label (some have more than others).
# - Generate fake data.
# 
# Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.
# 
# **NOTE:** The LeNet-5 implementation shown in the [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) at the end of the CNN lesson is a solid starting point. You'll have to change the number of classes and possibly the preprocessing, but aside from that it's plug and play!

# ### Implementation
# 
# Use the code cell (or multiple code cells, if necessary) to implement the first step of your project. Once you have completed your implementation and are satisfied with the results, be sure to thoroughly answer the questions that follow.

# In[ ]:

### Preprocess the data here.
### Feel free to use as many code cells as needed.


# In[5]:

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


# In[ ]:

'''
image = train['features'][label_dict[0][10]]
gray = rgb2gray(image)
print (image.shape, gray.shape)
plt.figure(figsize=[10,6])
plt.subplot(121)
plt.imshow(image)
plt.subplot(122)
plt.imshow(gray, cmap='gray')
'''


# In[6]:

# http://stackoverflow.com/questions/29831489/numpy-1-hot-array

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


# In[7]:

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


# In[8]:
def min_max_scale(image_data, low=0.1, high=0.9):
    return low + (image_data - np.min(image_data)) * (high - low) / (np.max(image_data) - np.min(image_data))

def LeNet(x, dropout):
    
    weights = {
        #'wc1': tf.Variable(tf.random_normal([5, 5, 1, 6])),
        #'wc2': tf.Variable(tf.random_normal([5, 5, 6, 16])),
        #'wd1': tf.Variable(tf.random_normal([5*5*16, 120])),
        #'out': tf.Variable(tf.random_normal([120, n_classes]))
        'wc1': tf.get_variable('wc1', shape=([5, 5, 1, 6]), initializer=tf.contrib.layers.xavier_initializer_conv2d()),
        'wc2': tf.get_variable('wc2', shape=([5, 5, 6, 16]), initializer=tf.contrib.layers.xavier_initializer_conv2d()),
        'wd1': tf.get_variable('wd1', shape=([5*5*16, 120]), initializer=tf.contrib.layers.xavier_initializer_conv2d()),
        'out': tf.get_variable('out', shape=([120, n_classes]), initializer=tf.contrib.layers.xavier_initializer_conv2d())
    }

    biases = {
        'bc1': tf.Variable(tf.zeros([6])),
        'bc2': tf.Variable(tf.zeros([16])),
        'bd1': tf.Variable(tf.zeros([120])),
        'out': tf.Variable(tf.zeros([n_classes]))
        #'bc1': tf.Variable(shape=([6]), initializer=tf.contrib.layers.xavier_initializer()),
        #'bc2': tf.Variable(shape=([16]), initializer=tf.contrib.layers.xavier_initializer()),
        #'bd1': tf.Variable(shape=([120]), initializer=tf.contrib.layers.xavier_initializer()),
        #'out': tf.Variable(shape=([n_classes]), initializer=tf.contrib.layers.xavier_initializer())
    }

    #dropout = 0.8

    x = tf.reshape(x, (-1, image_shape[0], image_shape[1], 1))
    
    # Conv Layer 1
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k=2)
    conv1 = tf.nn.dropout(conv1, dropout)

    # Conv Layer 2
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)
    conv2 = tf.nn.dropout(conv2, dropout)

    # FC Layer
    fc1 = tf.contrib.layers.flatten(conv2)
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    # Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Out Layer
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    
    return out


# In[9]:

# convert images to grayscale
features_train = rgb2gray(train['features'])
print(train['features'].shape, features_train.shape)
labels_train = train['labels']
print (labels_train.shape)

features_test = rgb2gray(test['features'])
print(test['features'].shape, features_test.shape)
labels_test = test['labels']
print (labels_test.shape)


# In[10]:

# flatten the images
features_train = features_train.reshape(len(features_train), -1)
features_test = features_test.reshape(len(features_test), -1)

# scale the images
features_train = min_max_scale(features_train)
features_test = min_max_scale(features_test)

# In[11]:

# split train set into training and validation
from sklearn.cross_validation import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(features_train, labels_train, test_size=0.2, random_state=42)
print (X_train.shape, y_train.shape, X_valid.shape, y_valid.shape)
#X_train = X_train.reshape(len(X_train), -1)
#X_valid = X_valid.reshape(len(X_valid), -1)
#print (X_train.shape, X_valid.shape)


# validation_data = {}
# validation_data['y'] = convertToOneHot(test['labels'])
# grayscale_images_test = rgb2gray(test['features']).reshape(n_test, -1)
# validation_data['x'] = grayscale_images_test
# validation_data['size'] = n_test

# In[12]:

y_train = convertToOneHot(y_train)
y_valid = convertToOneHot(y_valid)
labels_test = convertToOneHot(labels_test)


# one_hot_labels_train = convertToOneHot(train['labels'])
# one_hot_labels_test = convertToOneHot(test['labels'])
# print (train['labels'].shape, one_hot_labels_train.shape, test['labels'].shape, one_hot_labels_test.shape)

# In[13]:

# Create the Graph

keep_prob = 0.70
X = tf.placeholder(tf.float32, (None, image_shape[0]*image_shape[1]))
y = tf.placeholder(tf.float32, (None, n_classes))
fc2 = LeNet(X, keep_prob)
###
#prediction = tf.nn.softmax(fc2)
#cross_entropy = -tf.reduce_sum(y * tf.log(prediction + 1e-6), reduction_indices=1)
#loss = tf.reduce_mean(cross_entropy)
###

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(fc2, y))
opt = tf.train.AdamOptimizer(learning_rate=1.0e-3)
train_op = opt.minimize(loss_op)
correct_prediction = tf.equal(tf.argmax(fc2, 1), tf.argmax(y, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
wrong_prediction = tf.not_equal(tf.argmax(fc2, 1), tf.argmax(y, 1))


# In[14]:

# Feed dicts for training, validation, and test session
train_feed_dict = {X: X_train, y: y_train}
valid_feed_dict = {X: X_valid, y: y_valid}
#test_feed_dict = {features: X_test, labels: test_labels}


# def eval_data(dataset):
#     """
#     Given a dataset as input returns the loss and accuracy.
#     """
#     #steps_per_epoch = dataset.num_examples // BATCH_SIZE
#     steps_per_epoch = len(dataset[X]) // BATCH_SIZE
#     num_examples = steps_per_epoch * BATCH_SIZE
#     total_acc, total_loss = 0, 0
#     for step in range(steps_per_epoch):
#         batch_x, batch_y = dataset[X], dataset[y]
#         loss, acc = sess.run([loss_op, accuracy_op], feed_dict={X: batch_x, y: batch_y})
#         total_acc += (acc * batch_x.shape[0])
#         total_loss += (loss * batch_x.shape[0])
#     return total_loss/num_examples, total_acc/num_examples

# In[15]:

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy, total_loss = 0, 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        loss, accuracy =  sess.run([loss_op, accuracy_op], feed_dict={X: batch_x, y: batch_y})
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
        wrong =  sess.run(wrong_prediction, feed_dict={X: batch_x, y: batch_y})
        print (offset, len(batch_y[wrong].argmax(axis=1)))
        wrong_list = np.append(wrong_list, batch_y[wrong].argmax(axis=1))
    return wrong_list


# In[16]:

EPOCHS = 200
BATCH_SIZE = 128
LEARNING_RATE = 0.0005


# In[17]:

from time import time


# In[ ]:

# Measurements use for graphing loss and accuracy

#EPOCHS = 10
log_batch_step = 50
batches = []
loss_batch = []
train_acc_batch = []
valid_acc_batch = []
train_loss_batch = []
valid_loss_batch = []

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
                loss = sess.run(train_op, feed_dict={X: batch_x, y: batch_y})
                
                # Log every 50 batches
                '''
                if not step % log_batch_step:
                    # Calculate Training and Validation accuracy
                    training_accuracy, training_loss = sess.run([accuracy_op, loss_op], feed_dict=train_feed_dict)
                    validation_accuracy, validation_loss = sess.run([accuracy_op, loss_op], feed_dict=valid_feed_dict)
                    #print (i, step, training_accuracy, training_loss, validation_accuracy, validation_loss)
                    lo, pred = sess.run([loss_op, prediction], feed_dict={X: batch_x, y: batch_y})
                    print ('loss = ', lo)

                    print ('pred shape = ', pred.shape)
                    with tf.device('/gpu:0'):
                        ce = -tf.reduce_sum(batch_y * tf.log(pred + 1e-6), reduction_indices=1)
                    print ('ce shape = ', ce.get_shape())
                    print ('loss = ', lo)
                    ce_arr = ce.eval()
                    if (ce_arr < 0).any():
                        print (ce_arr)
                        for i in range(len(pred)):
                            print (pred[i])
                        assert 1==2
                    #print ('loss = ', loss)
                    
                # diagnosis
                #if (training_loss < 1000) and (training_accuracy < 0.1):
                #   assert 3==4
                #

                # Log batches
                previous_batch = batches[-1] if batches else 0
                batches.append(log_batch_step + previous_batch)
                train_acc_batch.append(training_accuracy)
                valid_acc_batch.append(validation_accuracy)
                train_loss_batch.append(training_loss)
                valid_loss_batch.append(validation_loss)
                '''
                
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
        print (sorted(zip(range(len(f)), f), key=lambda x: x[1]))
        print ("freq. wrong preds ", np.bincount(wrong_pred.astype(int)))
        print("Test loss = {:.3f}".format(test_loss))
        print("Test accuracy = {:.3f}".format(test_acc))
        
        # dump the trends into a file
        import pandas as pd
        trends = {'training_loss': train_loss_epoch, 'validation_loss': valid_loss_epoch,
                  'training_accuracy': train_acc_epoch, 'validation_accuracy': valid_acc_epoch}
        df_trend = pd.DataFrame(trends)
        df_trend.to_csv("trends.csv", index=False)

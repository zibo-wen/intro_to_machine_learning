%tensorflow_version 1.x

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the data
def loadData():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.0
        Target = Target[randIndx]
        trainData, trainTarget = Data[:10000], Target[:10000]
        validData, validTarget = Data[10000:16000], Target[10000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

# Implementation of a neural network using only Numpy - trained using gradient descent with momentum
def convertOneHot(trainTarget, validTarget, testTarget):
    newtrain = np.zeros((trainTarget.shape[0], 10))
    newvalid = np.zeros((validTarget.shape[0], 10))
    newtest = np.zeros((testTarget.shape[0], 10))

    for item in range(0, trainTarget.shape[0]):
        newtrain[item][trainTarget[item]] = 1
    for item in range(0, validTarget.shape[0]):
        newvalid[item][validTarget[item]] = 1
    for item in range(0, testTarget.shape[0]):
        newtest[item][testTarget[item]] = 1
    return newtrain, newvalid, newtest


def shuffle(trainData, trainTarget):
    np.random.seed(421)
    randIndx = np.arange(len(trainData))
    target = trainTarget
    np.random.shuffle(randIndx)
    data, target = trainData[randIndx], target[randIndx]
    return data, target


    
#https://www.datacamp.com/community/tutorials/cnn-tensorflow-python
epoch = 50 
learning_rate = 1e-04
batchSize = 32

# MNIST total classes (0-9 digits)
n_classes = 10

#Creating wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x) 

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')

'''
1. Input Layer
2. A 3 x 3 convolutional layer, with 32 filters, using vertical and horizontal strides of 1.
3. ReLU activation
4. A batch normalization layer
5. A 2 x 2 max pooling layer
6. Flatten layer
7. Fully connected layer (with 784 output units, i.e. corresponding to each pixel)
8. ReLU activation
9. Fully connected layer (with 10 output units, i.e. corresponding to each class)
10. Softmax output
11. Cross Entropy loss
'''
def conv_net(x, weights, biases, labels):  

    # 2 & 3
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    
    # 4
    mean, variance = tf.nn.moments(conv1, axes=[0, 1, 2])
    batchNormalization = tf.nn.batch_normalization(conv1, mean, variance, None, None, 1e-09)
    # 5
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 14*14 matrix.
    maxPoolingLayer = maxpool2d(batchNormalization)

    # 6 & 7 & 8
    fc1 = tf.reshape(maxPoolingLayer, [-1, weights['wc2'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wc2']), biases['bc2'])
    
    #2.3.2 dropout
    #fc1 = tf.nn.dropout(fc1,rate=0.1)
    #fc1 = tf.nn.dropout(fc1,rate=0.25)
    #fc1 = tf.nn.dropout(fc1,rate=0.5)
    
    fc1 = tf.nn.relu(fc1)
    
    #9
    fc2 = tf.reshape(fc1, [-1, weights['out'].get_shape().as_list()[0]])
    fc2 = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
    
    #10
    softmaxLayer = tf.nn.softmax(fc2)
    
    #11
    ceLost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels, softmaxLayer))
    
    #2.3.1 regularization
    ceLost = ceLost + 0*(tf.nn.l2_loss(weights['wc1']) + tf.nn.l2_loss(weights['wc2']) + tf.nn.l2_loss(weights['out']))
    
    
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ceLost);
    
    #Here you check whether the index of the maximum value of the predicted image is equal to the actual labelled image. and both will be a column vector.
    correct_prediction = tf.equal(tf.argmax(softmaxLayer, 1), tf.argmax(labels, 1))

    #calculate accuracy across all the given images and average them out. 
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    return ceLost, optimizer, accuracy

trainData, validData, testData, trainTarget, validTarget, testTarget = loadData();
# -1 means let the reshape decide on its own
trainData = trainData.reshape(-1, 28, 28, 1)
validData = validData.reshape(-1, 28, 28, 1)
testData = testData.reshape(-1, 28, 28, 1)
trainTarget, validTarget, testTarget = convertOneHot(trainTarget, validTarget, testTarget);

#both placeholders are of type float
x = tf.placeholder("float", [None, 28,28,1])
y = tf.placeholder("float", [None, n_classes])

weights = {
    'wc1': tf.get_variable('W0', shape=(3,3,1,32), initializer=tf.contrib.layers.xavier_initializer()), 
    'wc2': tf.get_variable('W1', shape=(14*14*32,784), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('W2', shape=(784,n_classes), initializer=tf.contrib.layers.xavier_initializer())
}
biases = {
    'bc1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'bc2': tf.get_variable('B1', shape=(784), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('B2', shape=(n_classes), initializer=tf.contrib.layers.xavier_initializer())
}

ceLost, optimizer, accuracy = conv_net(x, weights, biases, y);

init_op = tf.global_variables_initializer()

trainingError = []
validationError = []
testError = []
trainingAccuracy = []
validationAccuracy = []
testAccuracy = []

with tf.Session() as sess:
    sess.run(init_op)
    for i in range(epoch):
        trainData, trainTarget = shuffle(trainData, trainTarget);
        
        for j in range(int(len(trainData)/batchSize)):
            dataBatch = trainData[j*batchSize:(j+1)*batchSize]
            labelBatch = trainTarget[j*batchSize:(j+1)*batchSize]
            
            feed_dict={y: labelBatch,x: dataBatch}
            feed_dict_train={y: trainTarget,x: trainData}
            feed_dict_validation={y: validTarget,x: validData}
            feed_dict_test={y: testTarget,x: testData}
            
            sess.run(optimizer, feed_dict=feed_dict);
            
        trainingE, trainingA = sess.run([ceLost, accuracy], feed_dict=feed_dict_train);
        validationE, validationA = sess.run([ceLost, accuracy], feed_dict=feed_dict_validation);
        testE, testA = sess.run([ceLost, accuracy], feed_dict=feed_dict_test);

        if 1:
            trainingError.append(trainingE)
            validationError.append(validationE)
            testError.append(testE)
            trainingAccuracy.append(trainingA)
            validationAccuracy.append(validationA)
            testAccuracy.append(testA)

    plt.figure(1, figsize=(10, 10))
    trainingErrorLine, = plt.plot(trainingError, label='trainingError')
    validationErrorLine, = plt.plot(validationError, label='validationError')
    testErrorLine, = plt.plot(testError, label='testError')
    plt.ylabel('Error')
    plt.xlabel('Epoch')
    plt.title('Errors in Epoches')
    plt.legend([trainingErrorLine, validationErrorLine, testErrorLine], ['trainingError', 'validationError', 'testError'])

    plt.figure(2, figsize=(10, 10))
    trainingAccuracyLine, = plt.plot(trainingAccuracy, label='trainingAccuracy')
    validationAccuracyLine, = plt.plot(validationAccuracy, label='validationAccuracy')
    testAccuracyLine, = plt.plot(testAccuracy, label='testAccuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.title('Accuracy in Epoches')
    plt.legend([trainingAccuracyLine, validationAccuracyLine, testAccuracyLine], ['trainingAccuracy', 'validationAccuracy', 'testAccuracy'])

    plt.show()

    print('trainDataAccuracy',trainingAccuracy[-1])
    print('validDataAccuracy',validationAccuracy[-1])
    print('testDataAccuracy',testAccuracy[-1])
    print('trainDataError',trainingError[-1])
    print('validDataError',validationError[-1])
    print('testDataError',testError[-1])

import numpy as np
import matplotlib.pyplot as plt
import time

EPSILON = 1e-15

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

# Return x if x>0
def relu(x):
    return np.maximum(0, x)

# Subtract the maximum value of z from
# all its elements to prevent overflow
def softmax(x):
    exp_x = np.exp(x - np.nanmax(x, axis=0))
    return exp_x / np.sum(exp_x, axis=0)

# Prediction of a given layer
def computeLayer(X, W, b):
    return (W @ X) + b

# Cross Entropy Loss
def averageCE(target, prediction):
    N = target.shape[1]
    ce = -np.sum(target * np.log(prediction)) / N
    return ce

# Gradient of CE with respect to the softmax function outcome
def gradCE(target, prediction):
    return (prediction - target)

# Compute the predict and intermediate values for each of the
# layers in NN
def forwardPass(X, w_1, w_2, b_1, b_2):
    S_1 = computeLayer(X, w_1, b_1)
    x_1 = relu(S_1)
    S_2 = computeLayer(x_1, w_2, b_2)
    x_2 = softmax(S_2)
    return S_1, x_1, S_2, x_2

# Neural Network Training Process
def numpyNN(trainData, trainTarget, validData, validTarget, testData, testTarget, epochs, n_hidden, alpha, gamma):

    start = time.time()

    w_1 = np.random.randn(n_hidden, trainData.shape[0]) * (
            2 / (n_hidden + trainData.shape[0]))
    w_2 = np.random.randn(trainTarget.shape[0], n_hidden) * (
            2 / (trainTarget.shape[0] + n_hidden))
    b_1 = np.zeros((n_hidden, 1))
    b_2 = np.zeros((trainTarget.shape[0], 1))

    # initialize parameter weights
    # w_1, w_2, b_1, b_2 = loadWeights(trainData.T, trainTarget.T, n_hidden)

    # momentum parameters
    v_1 = np.full((n_hidden, trainData.shape[0]), EPSILON)
    vb_1 = np.full((n_hidden, 1), EPSILON)
    v_2 = np.full((trainTarget.shape[0], n_hidden), EPSILON)
    vb_2 = np.full((trainTarget.shape[0], 1), EPSILON)

    # For plotting error per iteration
    train_error = []
    valid_error = []
    test_error = []
    train_acc = []
    valid_acc = []
    test_acc = []

    for i in range(1, epochs):
        S_1, x_1, S_2, x_2 = forwardPass(trainData, w_1, w_2, b_1, b_2)
        train_error.append(averageCE(trainTarget, x_2))
        train_acc.append(calculateAccuraccy(trainTarget, x_2))

        _, _, _, valid_y_hat = forwardPass(validData, w_1, w_2, b_1, b_2)
        valid_error.append(averageCE(validTarget, valid_y_hat))
        valid_acc.append(calculateAccuraccy(validTarget, valid_y_hat))

        _, _, _, test_y_hat = forwardPass(testData, w_1, w_2, b_1, b_2)
        test_error.append(averageCE(testTarget, test_y_hat))
        test_acc.append(calculateAccuraccy(testTarget, test_y_hat))

        # --- Back propagation ---
        N = trainData.shape[1]

        # Calculate gradients
        dE_dS2 = gradCE(trainTarget, x_2)
        dE_dw_2 = dE_dS2 @ x_1.T / N
        dE_db_2 = np.sum(dE_dS2, 1, keepdims=True) / N
        dE_dS1 = w_2.T @ dE_dS2 * (x_1 > 0)
        dE_dw_1 = dE_dS1 @ trainData.T / N
        dE_db_1 = np.sum(dE_dS1, 1, keepdims=True) / N

        # Calculate new momentum
        v_1 = gamma * v_1 + alpha * dE_dw_1
        v_2 = gamma * v_2 + alpha * dE_dw_2
        vb_1 = gamma * vb_1 + alpha * dE_db_1
        vb_2 = gamma * vb_2 + alpha * dE_db_2

        # Calculate new weights and bias
        w_1 = w_1 - v_1
        w_2 = w_2 - v_2
        b_1 = b_1 - vb_1
        b_2 = b_2 - vb_2

    print('The training took ' + str(time.time() - start) + ' seconds. ')

    printErr(train_error, valid_error, test_error, 'Errors with Hidden Unites #' + str(n_hidden))
    printAcc(train_acc, valid_acc, test_acc, 'Accuracy with Hidden Unites #' + str(n_hidden))

    # Accuracies
    print("Final Training Accuracy = ", train_acc[-1])
    print("Final Validation Accuracy = ", valid_acc[-1])
    print("Final Test Accuracy = ", test_acc[-1])

    # For Early Stopping
    print("Max Validation Accuracy is: ", max(valid_acc))
    maxIndex = valid_acc.index(max(valid_acc))
    print("In epoch #: ", maxIndex)
    print("\nTraining Accuracy at early stop point is ", train_acc[maxIndex])
    print("Validation Accuracy at early stop point is ", valid_acc[maxIndex])
    print("Test Accuracy at early stop point is ", test_acc[maxIndex])


def calculateAccuraccy(y, y_hat):
    predictions = np.argmax(y_hat, 0)
    labels = np.argmax(y, 0)
    acc = np.sum(predictions == labels) / y.shape[1]
    return acc

# Plot error
def printErr(train_error, valid_error, test_error, title):
    plt.figure(figsize=(5, 5))
    plt.plot(train_error, label="Training")
    plt.plot(valid_error, label="Validation")
    plt.plot(test_error, label="Testing")
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.grid()
    plt.title(title)
    plt.legend(loc='upper right')
    #plt.savefig('/img/' + title + '.png')
    plt.show()
    plt.clf()


# Plot accuracy
def printAcc(train_acc, valid_acc, test_acc, title):
    plt.figure(figsize=(5, 5))
    plt.plot(train_acc, label="training accuracy")
    plt.plot(valid_acc, label="validation accuracy")
    plt.plot(test_acc, label="testing accuracy")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.title(title)
    plt.legend(loc='lower right')
    #plt.savefig('/img/' + title + '.png')
    plt.show()
    plt.clf()


def npNNRun():
    # Load data and reshape
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    trainData = trainData.reshape(trainData.shape[0], trainData.shape[1] * trainData.shape[2])
    validData = validData.reshape(validData.shape[0], validData.shape[1] * validData.shape[2])
    testData = testData.reshape(testData.shape[0], testData.shape[1] * testData.shape[2])

    # Convert into One-Hot
    trainTarget, validTarget, testTarget = convertOneHot(trainTarget, validTarget, testTarget)

    n_hiddenList = [100, 500, 1000, 2000]
    for n_hidden in n_hiddenList:
        numpyNN(trainData.T, trainTarget.T, validData.T, validTarget.T, testData.T, testTarget.T,
                epochs=200, n_hidden=n_hidden, alpha=0.05, gamma=0.99)

if __name__ == '__main__':
    npNNRun()

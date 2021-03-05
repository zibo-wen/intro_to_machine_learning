%tensorflow_version 1.x

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp

print(tf.__version__)

# Loading data
#data = np.load('data2D.npy')
data = np.load('data100D.npy')
[num_pts, dim] = np.shape(data)

is_valid = 1;

# For Validation set
if is_valid:
  valid_batch = int(num_pts / 3.0)
  np.random.seed(45689)
  rnd_idx = np.arange(num_pts)
  np.random.shuffle(rnd_idx)
  val_data = data[rnd_idx[:valid_batch]]
  data = data[rnd_idx[valid_batch:]]
    
[num_pts, dim] = np.shape(data)


# Distance function for K-means
def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the squared pairwise distance matrix (NxK)
    N = X.get_shape().as_list()[0];
    K = MU.get_shape().as_list()[0];
    X_repeat = tf.repeat(X,repeats=K,axis=0);
    MU_tile = tf.tile(MU, [N,1]);
    reducedSum = tf.reduce_sum(tf.square(X_repeat-MU_tile),1);
    pair_dist = tf.reshape(reducedSum,[-1,K]);
    return pair_dist;

def validationLoss(MU):
    N = val_data.shape[0];
    K = MU.shape[0];
    val_data_repeat = np.repeat(val_data,repeats=K,axis=0);
    MU_tile = np.tile(MU, [N,1]);
    reducedSum = np.sum(np.square(val_data_repeat-MU_tile),1);
    pair_dist = np.reshape(reducedSum,[-1,K]);
    validLoss = np.sum(np.amin(pair_dist, axis=1));
    return validLoss;

def K_means(numClusters):
    print("\n\nNow Running with K =", numClusters)  
    MU = tf.Variable(tf.truncated_normal(shape=(numClusters, dim), stddev=0.5, dtype=tf.float32))
    X = tf.placeholder(tf.float32, shape=(num_pts, dim))

    pair_dist = distanceFunc(X, MU);
    loss = tf.reduce_sum(tf.reduce_min(pair_dist, axis=1));

    optimizer = tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(loss)

    init_op = tf.global_variables_initializer()

    trainingError = []
    validationError = []
    colors = []
    epoch = 500
    cluster1 = 0;
    cluster2 = 0;
    cluster3 = 0;
    cluster4 = 0;
    cluster5 = 0;
    distanceMatrix = 0;

    with tf.Session() as sess:
        sess.run(init_op)
        feed_dict={X: data}
        for i in range(epoch):
            distanceMatrix, trainingE, _ = sess.run([pair_dist, loss, optimizer], feed_dict=feed_dict);
            if (is_valid):
                validationError.append(validationLoss(MU.eval()));

            
            trainingError.append(trainingE);

        '''
        plt.figure(1, figsize=(10, 10))
        plt.plot(trainingError)
        plt.ylabel('The loss')
        plt.xlabel('The number of updates')
        plt.title('The loss vs the number of updates')

        argminDistance = np.argmin(distanceMatrix, axis=1);
        for i in range(num_pts):
            if (argminDistance[i]==0):
                colors.append('blue');
                cluster1+=1;
            elif (argminDistance[i]==1):
                colors.append('green');
                cluster2+=1;
            elif (argminDistance[i]==2):
                colors.append('red');
                cluster3+=1;
            elif (argminDistance[i]==3):
                colors.append('yellow');
                cluster4+=1;
            elif (argminDistance[i]==4):
                colors.append('orange');
                cluster5+=1;

        plt.figure(2, figsize=(10, 10))
        plt.scatter(data[:,0], data[:,1], c=colors);
        plt.scatter(MU.eval()[:,0], MU.eval()[:,1], c='black');

        if (is_valid):
            plt.figure(3, figsize=(10, 10))
            plt.plot(validationError)
            plt.ylabel('The validation loss')
            plt.xlabel('The number of updates')
            plt.title('The validation loss vs the number of updates')
        
        plt.show()

        print('cluster 1 contains',cluster1/num_pts*100,'% of the points');
        print('cluster 2 contains',cluster2/num_pts*100,'% of the points');
        print('cluster 3 contains',cluster3/num_pts*100,'% of the points');
        print('cluster 4 contains',cluster4/num_pts*100,'% of the points');
        print('cluster 5 contains',cluster5/num_pts*100,'% of the points');
        '''

        print('trainDataError',trainingError[-1]);
        if (is_valid):
            print('validationDataError',validationError[-1]);
            print('trainDataError to validationDataError ratio: ', validationError[-1]/trainingError[-1])
            return validationError
'''        
K_means(1);
K_means(2);
K_means(3);
K_means(4);
K_means(5);
'''

valdErr1 = K_means(5)
valdErr2 = K_means(10)
valdErr3 = K_means(15)
valdErr4 = K_means(20)
valdErr5 = K_means(30)

plt.figure(3, figsize=(10, 10))
plt.plot(valdErr1, 'r', label="K=5")
plt.plot(valdErr2, 'b', label="K=10")
plt.plot(valdErr3, label="K=15")
plt.plot(valdErr4, label="K=20")
plt.plot(valdErr5, label="K=30")

plt.ylabel('The validation loss')
plt.xlabel('The number of updates')
plt.title('The validation loss vs the number of updates (K-means)')
plt.legend()

%tensorflow_version 1.x

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp

dataD = 2

# Loading data
if dataD == 100:
    data = np.load('data100D.npy')
else:
    data = np.load('data2D.npy')
[num_pts, dim] = np.shape(data)

is_valid = False

# For Validation set
if is_valid:
  valid_batch = int(num_pts / 3.0)
  np.random.seed(45689)
  rnd_idx = np.arange(num_pts)
  np.random.shuffle(rnd_idx)
  val_data = data[rnd_idx[:valid_batch]]
  data = data[rnd_idx[valid_batch:]]

[num_pts, dim] = np.shape(data)

# Distance function for GMM
def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the pairwise distance matrix (NxK)
    # TODO
    N = X.get_shape().as_list()[0];
    K = MU.get_shape().as_list()[0];
    X_repeat = tf.repeat(X,repeats=K,axis=0);
    MU_tile = tf.tile(MU, [N,1]);
    reducedSum = tf.reduce_sum(tf.square(X_repeat-MU_tile),1);
    pair_dist = tf.reshape(reducedSum,[-1,K]) #tf.transpose(reducedSum)
    return pair_dist;

#assume sigma is already squared, or else need to fix it
def log_GaussPDF(X, mu, sigma):
    # Inputs
    # X: N X D
    # mu: K X D
    # sigma: K X 1

    # Outputs:
    # log Gaussian PDF N X K

    # TODO
    N = X.get_shape().as_list()[0];
    pair_dist = distanceFunc(X,mu)
    sigma_repeat = tf.repeat(tf.transpose(sigma),repeats = N, axis=0)
    result = -(dim/2)*tf.log(2*np.pi*sigma_repeat) - (pair_dist)/(2*sigma_repeat)
    return result;

def log_posterior(log_PDF, log_pi):
    # Input
    # log_PDF: log Gaussian PDF N X K
    # log_pi: K X 1

    # Outputs
    # log_post: N X K

    log_pi = tf.squeeze(log_pi)
    return log_PDF + log_pi - hlp.reduce_logsumexp(log_PDF + log_pi, 1, True)


def MoG(numClusters):
    print("\n\nNow Running with K =", numClusters)   
    MU = tf.Variable(tf.truncated_normal(shape=(numClusters, dim), stddev=0.5, dtype=tf.float32))
    X = tf.placeholder(tf.float32, shape=(num_pts, dim))
    phi = tf.Variable(tf.truncated_normal(shape=(numClusters, 1), dtype=tf.float32))
    psi = tf.Variable(tf.truncated_normal(shape=(numClusters, 1), dtype=tf.float32)) 

    sig_sqrt = tf.exp(phi)
    log_pi = hlp.logsoftmax(psi)

    log_gauss_val = log_GaussPDF(X, MU, sig_sqrt)
    log_post = log_posterior(log_gauss_val, log_pi)


    loss = - tf.reduce_sum(hlp.reduce_logsumexp(log_gauss_val + tf.transpose(log_pi), 1, keep_dims=True), axis=0)

    if (is_valid):
        V = tf.placeholder(tf.float32, shape=(np.shape(val_data)))
        val_log_gauss_val = log_GaussPDF(V, MU, sig_sqrt)
        v_loss =  -tf.reduce_sum(hlp.reduce_logsumexp(val_log_gauss_val + tf.transpose(log_pi), 1), axis=0)

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
            MU_val, trainingE, sig_sqrt_val, log_gauss_val_val, log_pi_val, log_post_val, _ = sess.run(
                [MU, loss, sig_sqrt, log_gauss_val, log_pi, log_post, optimizer], feed_dict=feed_dict)
            #print(tf.count_nonzero(tf.is_nan(MU_val)).eval())
            #print(MU.eval())
            if (is_valid):
                val_err = sess.run(v_loss, feed_dict={V: val_data})
                validationError.append(val_err)

            if 1:
                trainingError.append(trainingE)
                
        print("Mean:")
        print(MU.eval())
        print("pi:")
        print(log_pi_val)
        print("sig:")
        print(sig_sqrt_val)
        

        plt.figure(1, figsize=(10, 10))
        plt.plot(trainingError)
        plt.ylabel('The loss')
        plt.xlabel('The number of updates')
        plt.title('The loss vs the number of updates')
  
        
        if dataD == 2:
            cluster_pred = np.argmax(log_post_val, axis=1) 
            plt.figure(2, figsize=(10, 10))
            plt.scatter(data[:,0], data[:,1], c=cluster_pred);
            plt.scatter(MU.eval()[:,0], MU.eval()[:,1], c='red', marker='X');
        
        if (is_valid):
            plt.figure(3, figsize=(10, 10))
            plt.plot(validationError)
            plt.ylabel('The validation loss')
            plt.xlabel('The number of updates')
            plt.title('The validation loss vs the number of updates')

        for i in range(numClusters):
            print('cluster', i, 'contains', 
                np.count_nonzero(cluster_pred == i)/num_pts*100,
                '% of the points');
        
        plt.show()
        
        print('trainDataError',trainingError[-1]);
        if (is_valid):
            print('validationDataError',validationError[-1]);
            print('trainDataError to validationDataError ratio: ',
                      validationError[-1]/trainingError[-1])
        

        return validationError

if __name__ == '__main__':

    if dataD == 2:
        # Training with K=1,2,3,4,5 for the 2D dataset, as the instruction requires
        valdErr1 = MoG(1)
        valdErr2 = MoG(2)
        valdErr3 = MoG(3)
        valdErr4 = MoG(4)
        valdErr5 = MoG(5)


        plt.figure(3, figsize=(10, 10))
        plt.plot(valdErr1, label="K=1")
        plt.plot(valdErr2, label="K=2")
        plt.plot(valdErr3, label="K=3")
        plt.plot(valdErr4, label="K=4")
        plt.plot(valdErr5, label="K=5")
        plt.ylabel('The validation loss')
        plt.xlabel('The number of updates')
        plt.title('The validation loss vs the number of updates (MoG)')
        plt.legend()


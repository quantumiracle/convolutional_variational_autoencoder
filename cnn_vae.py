from __future__ import division
import argparse
import matplotlib.pyplot as plt
import pickle
import gzip
import numpy as np
import tensorflow as tf
import matplotlib.gridspec as gridspec
import os
from tensorflow.examples.tutorials.mnist import input_data
#np.set_printoptions(threshold=np.inf)

f =gzip.open('./data_space_8080.gzip','rb')
save_file='./model.ckpt'

mb_size = 100
z_dim = 100
#X_dim = 210*160
X_dim = 80
conv_dim = 10
h_dim = 128
c = 0
#lr = 1e-4

'''
def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig
'''
def lrelu(x, alpha=0.1):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


# =============================== Q(z|X) ======================================

X = tf.placeholder(tf.float32, shape=[X_dim*X_dim])
z = tf.placeholder(tf.float32, shape=[None, z_dim])
lr = tf.placeholder(tf.float32)


Q_W1 = tf.Variable(xavier_init([int(X_dim*X_dim/(2*2)*conv_dim), h_dim]))
Q_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

Q_W2_mu = tf.Variable(xavier_init([h_dim, z_dim]))
Q_b2_mu = tf.Variable(tf.zeros(shape=[z_dim]))

Q_W2_sigma = tf.Variable(xavier_init([h_dim, z_dim]))
Q_b2_sigma = tf.Variable(tf.zeros(shape=[z_dim]))



def Q(X):
    X = tf.reshape(X, [-1, X_dim, X_dim, 1])
    conv = tf.contrib.layers.conv2d(X,
                                    conv_dim,
                                    [5, 5],
                                    (2, 2),
                                    padding='SAME',
                                    activation_fn=lrelu,
                                    normalizer_fn=tf.contrib.layers.batch_norm)
    flat = tf.contrib.layers.flatten(conv)
    #print(flat.shape)
    h = tf.nn.relu(tf.matmul(flat, Q_W1) + Q_b1)
    z_mu = tf.matmul(h, Q_W2_mu) + Q_b2_mu
    z_logvar = tf.matmul(h, Q_W2_sigma) + Q_b2_sigma
    return z_mu, z_logvar


def sample_z(mu, log_var):
    eps = tf.random_normal(shape=tf.shape(mu))
    return mu + tf.exp(log_var / 2) * eps


# =============================== P(X|z) ======================================

P_W1 = tf.Variable(xavier_init([z_dim, h_dim]))
P_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

P_W2 = tf.Variable(xavier_init([h_dim, int(X_dim*X_dim/(2*2)*conv_dim)]))
P_b2 = tf.Variable(tf.zeros(shape=[int(X_dim*X_dim/(2*2)*conv_dim)]))


def P(z):
    h = tf.nn.relu(tf.matmul(z, P_W1) + P_b1)
    logits = tf.matmul(h, P_W2) + P_b2
    logits=tf.reshape(logits, [-1,int(X_dim/2),int(X_dim/2),conv_dim])
    trans_conv = tf.contrib.layers.conv2d_transpose(logits,
                                                    1,
                                                    [5, 5],
                                                    (2, 2),
                                                    padding='SAME',
                                                    activation_fn=lrelu,
                                                    normalizer_fn=tf.contrib.layers.batch_norm)

    out = tf.nn.relu(trans_conv)
    return out, logits


# =============================== TRAINING ====================================
saver = tf.train.Saver()

z_mu, z_logvar = Q(X)
z_sample = sample_z(z_mu, z_logvar)
out, logits = P(z_sample)

# Sampling from random z
X_samples, _ = P(z)

# E[log P(X|z)]
recon_loss = tf.reduce_sum(tf.abs(out -  X))
# D_KL(Q(z|X) || P(z)); calculate in closed form as both dist. are Gaussian
kl_loss = 0.5 * tf.reduce_sum(tf.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1)
#recon_loss=tf.reduce_sum(tf.abs(X -  X))

# VAE loss
vae_loss = tf.reduce_mean(recon_loss + kl_loss)

solver = tf.train.AdamOptimizer(lr).minimize(vae_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('convae_space/'):
    os.makedirs('convae_space/')

k = 0
Loss=[]
It=[]
train_times=10000
for it in range(train_times):

    batch=pickle.load(f)/100


    #print(batch)
    '''
    print(batch[0])
    plt.imshow(batch[0].reshape(80,80))
    plt.show()
    '''
    #print(batch)
    if it < 7000:
        _, loss ,recon_l, kl_l, output = sess.run([solver, vae_loss,recon_loss,kl_loss,out], feed_dict={X: batch,lr:1e-3})
    else:
        _, loss ,recon_l, kl_l, output = sess.run([solver, vae_loss,recon_loss,kl_loss,out], feed_dict={X: batch,lr:1e-6})

    Loss.append(loss)
    It.append(it)
    
    if it % 100 == 0:
        print('Iter: {}'.format(it))
        #print('Loss: {:.4}'. format(loss),recon_l,kl_l)
        print('Loss: {:.4}'. format(loss))

        #Z_sample = sess.run([z_sample], feed_dict={X: batch})
        #print('z_sample: ',Z_sample[0][1])
        samples = sess.run([X_samples], feed_dict={z: np.random.randn(1,z_dim)})


        for j,sample in enumerate(samples):
            print(len(sample))
            plt.imshow(sample.reshape(80,80))
            #plt.imshow(sample.reshape(210,160))
        
            #fig = plot(samples)
        plt.savefig('convae_space/{}.png'.format(str(k).zfill(3)), bbox_inches='tight')
        k += 1
saver.save(sess, save_file)
fig,ax = plt.subplots() 
ax.plot(It,Loss,color='blue')
#plt.ylim(0,50)
fig.set_size_inches(12,6)

#plt.savefig('com.png')
plt.show()
f.close()
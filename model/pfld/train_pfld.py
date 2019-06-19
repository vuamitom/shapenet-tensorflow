# from shapnet import predict_landmarks as shapenet_predict_landmarks
from pfld import predict_landmarks as pfld_predict_landmarks
import numpy as np
import tensorflow as tf
import math
import sys

BATCH_SIZE = 20
NO_EPOCH = 1000
IMAGE_SIZE = 224

def normalize_data(data):
    return (data - 0.5) * 2

class DataSet:
    def __init__(self, path, batch_size):
        with np.load(path) as ds:
            # ds = np.load(path)
            self.data = ds['data']
            # normalize data
            self.data = normalize_data(self.data)# (self.data - 0.5) * 2
            self.labels = ds['labels']
            self.labels = self.labels.reshape((-1, 1, 136)).squeeze()
        self.idx = 0
        self.batch_size = batch_size

    def reset_and_shuffle(self):
        permutation = np.random.permutation(self.labels.shape[0])
        self.data = self.data[permutation,:,:,:]
        self.labels = self.labels[permutation]
        self.idx = 0    

    def no_batches(self):
        return math.floor(self.size() * 1.0/ self.batch_size)

    def size(self):
        return self.data.shape[0]

    def next_batch(self):
        if self.idx >= self.size():
            return None, None
        n = self.idx + self.batch_size
        n = n if n < self.size() else self.size()
        data = self.data[self.idx:n, :]
        labels = self.labels[self.idx:n, :]
        self.idx = n
        return data, labels


def train(data_path, save_path,
                            image_size=IMAGE_SIZE,
                            batch_size=BATCH_SIZE,
                            no_epoch=NO_EPOCH,
                            checkpoint=None, 
                            quantize=True,
                            quant_delay=50000,
                            step_per_save=300,
                            lr=0.001):

    print('train with image_size ', image_size, 
        ' quantize=', quantize, 
        'batch_size=', batch_size,         
        'learning rate = ', lr)

    inputs = tf.placeholder(tf.float32, shape=[None, image_size, image_size, 3], name='input_images')
    labels = tf.placeholder(tf.float32, shape=[None, 136], name='landmarks')

    preds = pfld_predict_landmarks(inputs,
                                is_training=True)
    # define loss function
    
    l1_loss = tf.losses.absolute_difference(labels, preds)
    mse_loss = tf.losses.mean_squared_error(labels, preds)

    if quantize:
        print('add custom op for quantize aware training after delay', quant_delay)
        tf.contrib.quantize.create_training_graph(input_graph=tf.get_default_graph(), quant_delay=quant_delay)
    global_step = tf.train.get_or_create_global_step()    
    optimizer = tf.train.AdamOptimizer(lr, 0.9, 0.999)

    # refer to https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/layers.py#L473
    update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        print('add dependency on "moving avg" for batch_norm')        
        train_op = optimizer.minimize(l1_loss, global_step) 
    
    ds = DataSet(data_path, batch_size)

    saver = tf.train.Saver()
    with tf.Session() as sess: 
        if checkpoint:
            print('restore from  checkpoint', checkpoint)
            saver.restore(sess, checkpoint)
        else:
            print('train from scratch')
            sess.run(tf.global_variables_initializer())

        for epoch in range(0, no_epoch):
            # sess.run(iterator.initializer, )
            ds.reset_and_shuffle()
            for batch_no in range(0, ds.no_batches()):
                train_data, train_labels = ds.next_batch()        
                # train_data = (train_data - 0.5) * 2 
                _, l1_loss_val, mse_loss_val, step = sess.run([train_op, l1_loss, mse_loss, global_step], feed_dict={
                    inputs: train_data, labels: train_labels
                })
                if step > 0 and step % 100 == 0:
                    print('step ', step, 'l1 loss = ', l1_loss_val, 'mse_loss = ', mse_loss_val)
                    if step % step_per_save == 0:
                        # save
                        result_path = saver.save(sess, save_path, global_step=step)
                        print ('saved to ', result_path)
            # print('end of epoch, eval model')

if __name__ == '__main__':
    train('../../data/labels_ibug_300W_train_64.npz', 
        '../../data/checkpoints-pfld-64/shapenet',
        checkpoint='../../data/checkpoints-pfld-64/pfld-218400',
        image_size=64,
        quantize=False, lr=0.001) 
    # else:
    #     train('../data/labels_ibug_300W_train_112_grey.npz', 
    #         '../data/unrot_train_pca.npz',
    #         '../data/checkpoints-112-1/shapenet',
    #         image_size=112,
    #         in_channels=1,
    #         extractor=extractors.custom_feature_extractor,
    #         quantize=False, lr=0.001)
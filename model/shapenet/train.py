from shapnet import predict_landmarks as shapenet_predict_landmarks
# from pfld import predict_landmarks as pfld_predict_landmarks
import numpy as np
import tensorflow as tf
import math
import sys
import extractors
N_COMPONENTS = 12
BATCH_SIZE = 20
NO_EPOCH = 1000
IMAGE_SIZE = 224

class DataSet:
    def __init__(self, path, batch_size, in_channels=1):
        with np.load(path) as ds:
            # ds = np.load(path)
            self.data = ds['data']
            # normalize data
            # self.data = (self.data - 0.5) * 2
            self.labels = ds['labels']
        self.idx = 0
        self.in_channels = in_channels
        self.batch_size = batch_size

    def reset_and_shuffle(self):
        permutation = np.random.permutation(self.labels.shape[0])
        self.data = self.data[permutation,:,:,:] if self.in_channels == 3 else self.data[permutation,:,:]
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

def get_pcomps(pca_path):
    return np.load(pca_path)['shapes'][:(N_COMPONENTS + 1)] 

def train(data_path, pca_path, save_path,
                            image_size=IMAGE_SIZE,
                            batch_size=BATCH_SIZE,
                            no_epoch=NO_EPOCH,
                            checkpoint=None, 
                            quantize=True,
                            in_channels=1,
                            extractor=extractors.original_paper_feature_extractor,
                            quant_delay=50000,
                            step_per_save=300,
                            lr=0.001):

    print('train with image_size ', image_size, 
        ' quantize=', quantize, 
        'batch_size=', batch_size, 
        'extractor=', extractor,
        'in_channels=', in_channels,
        'learning rate = ', lr)
    components = get_pcomps(pca_path)

    # define input_variables
    input_shape = [None, image_size, image_size] if in_channels == 1 else [None, image_size, image_size, 3]
    inputs = tf.placeholder(tf.float32, shape=input_shape, name='input_images')
    labels = tf.placeholder(tf.float32, shape=[None, 68, 2], name='landmarks')

    preds = shapenet_predict_landmarks(inputs, components, 
                                is_training=True, 
                                feature_extractor=extractor)
    
    # define loss function
    
    l1_loss = tf.losses.absolute_difference(labels, preds)
    mse_loss = tf.losses.mean_squared_error(labels, preds)

    if quantize:
        print('add custom op for quantize aware training after delay', quant_delay)
        tf.contrib.quantize.create_training_graph(input_graph=tf.get_default_graph(), quant_delay=quant_delay)
    global_step = tf.train.get_or_create_global_step()
    # tf.summary.scalar('losses/l1_loss', l1_loss)
    # tf.summary.scalar('losses/mse_loss', mse_loss)
    # summary_op = tf.summary.merge_all()
    # define optimizer    
    optimizer = tf.train.AdamOptimizer(lr, 0.9, 0.999)
    train_op = optimizer.minimize(l1_loss, global_step) 
    # train_data, train_labels = None, None
    # with np.load(data_path) as np_data:
    #     train_data = np_data['data']
    #     train_labels = np_data['labels']
    ds = DataSet(data_path, batch_size, in_channels=in_channels)

    saver = tf.train.Saver()
    with tf.Session() as sess: 
        if checkpoint:
            print('restore from  checkpoint', checkpoint)
            saver.restore(sess, checkpoint)
        else:
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
            print ('end of epoch')

if __name__ == '__main__':
    train('../../data/labels_ibug_300W_train_224_grey.npz', 
        '../../data/unrot_train_pca.npz',
        '../../data/checkpoints/shapenet',
        checkpoint=None,
        image_size=224,
        in_channels=1,
        quantize=True, lr=0.001) 
    # else:
    #     train('../data/labels_ibug_300W_train_112_grey.npz', 
    #         '../data/unrot_train_pca.npz',
    #         '../data/checkpoints-112-1/shapenet',
    #         image_size=112,
    #         in_channels=1,
    #         extractor=extractors.custom_feature_extractor,
    #         quantize=False, lr=0.001)
from shapnet import predict_landmarks
import numpy as np
import tensorflow as tf
import math
import sys

N_COMPONENTS = 12
BATCH_SIZE = 32
NO_EPOCH = 1000
IMAGE_SIZE = 224

class DataSet:
    def __init__(self, path, batch_size):
        with np.load(path) as ds:
            # ds = np.load(path)
            self.data = ds['data']
            # normalize image data - first verify the value range
            # self.data = (self.data - 127.5) / 128.0
            # print(self.data[0])
            # print(self.data[100])
            # print(self.data[200])
            self.labels = ds['labels']
        self.idx = 0
        self.batch_size = batch_size

    def reset_and_shuffle(self):
        permutation = np.random.permutation(self.labels.shape[0])
        self.data = self.data[permutation,:,:]
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
                            extractor='custom',
                            lr=0.001):
    
    components = get_pcomps(pca_path)

    # define input_variables
    input_shape = [None, image_size, image_size] if in_channels == 1 else [None, image_size, image_size, 3]
    inputs = tf.placeholder(tf.float32, shape=input_shape, name='input_images')
    labels = tf.placeholder(tf.float32, shape=[None, 68, 2], name='landmarks')

    # dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))    
    # dataset = dataset.batch(batch_size)
    # dataset = dataset.shuffle(buffer_size=10000)
    # iterator = dataset.make_initializable_iterator()
    # next_batch, next_label = iterator.get_next()
    # print('next batch ', next_batch)
    # print('next label ', next_label)

    preds = predict_landmarks(inputs, components, 
                                is_training=True, 
                                extractor=extractor)
    
    # define loss function
    
    l1_loss = tf.losses.absolute_difference(labels, preds)
    mse_loss = tf.losses.mean_squared_error(labels, preds)

    if quantize:
        print('add custom op for quantize aware training')
        tf.contrib.quantize.create_training_graph(input_graph=tf.get_default_graph(), quant_delay=50000)
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
    ds = DataSet(data_path, batch_size)
    print('Done load data')

    saver = tf.train.Saver()
    with tf.Session() as sess: 
        if checkpoint:
            saver.restore(sess, checkpoint)
        else:
            sess.run(tf.global_variables_initializer())

        for epoch in range(0, no_epoch):
            # sess.run(iterator.initializer, )
            ds.reset_and_shuffle()
            for batch_no in range(0, ds.no_batches()):
                train_data, train_labels = ds.next_batch()        
                train_data = (train_data - 0.5) * 2 
                _, l1_loss_val, mse_loss_val, step = sess.run([train_op, l1_loss, mse_loss, global_step], feed_dict={
                    inputs: train_data, labels: train_labels
                })
            
                if step > 0 and step % 100 == 0:
                    print('step ', step, 'l1 loss = ', l1_loss_val, 'mse_loss = ', mse_loss_val)
                    if step % 200 == 0:
                        # save
                        result_path = saver.save(sess, save_path, global_step=step)
                        print ('saved to ', result_path)
            print ('end of epoch')


    # with tf.Session() as sess:        
    #     try:
    #         # todo restore if ncessary
    #         if checkpoint:
    #             saver.restore(sess, checkpoint)
    #         else:
    #             init = tf.global_variables_initializer()
    #             sess.run(init)
    #         # coord = tf.train.Coordinator()
    #         # threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #         for epoch in range(0, no_epoch):
    #             # TODO: shuffle dataset 
    #             # if coord.should_stop():
    #             #     break
    #             ds.reset_and_shuffle()
    #             total_l1_loss = 0
    #             total_mse_loss = 0
    #             global_step_val = 0
    #             for batch_no in range(0, ds.no_batches()):        
    #                 # if coord.should_stop():
    #                 #     break
    #                 batch_input, batch_label = ds.next_batch()
    #                 _, summary, l1_loss_val, mse_loss_val = sess.run([train_op, summary_op, l1_loss, mse_loss], 
    #                                                         feed_dict={inputs: batch_input, labels: batch_label}) 
    #                 total_l1_loss += l1_loss_val
    #                 total_mse_loss += mse_loss_val   
    #                 global_step_val = sess.run(global_step)
    #                 if global_step_val % 200 == 0:
    #                     result_path = saver.save(sess, save_path, global_step=global_step_val)
    #                     print('save to', result_path, ': l1 loss = ', l1_loss_val, 'mse_loss = ', mse_loss_val)

    #             print('epoch ', epoch, '/', no_epoch, ': avg. l1 loss=', 
    #                 total_l1_loss/ds.no_batches(), ' avg. mse loss = ', 
    #                 total_mse_loss/ ds.no_batches())
    #             result_path = saver.save(sess, save_path, global_step=global_step_val)
    #             print('save end of epoch to ', result_path)

    #     finally:
    #         pass
            # coord.request_stop()
            # coord.join(threads)

if __name__ == '__main__':
    if True:
        train('../data/labels_ibug_300W_train_64.npz', 
            '../data/unrot_train_pca.npz',
            '../data/checkpoints/shapenet',
            image_size=64,
            in_channels=3,
            extractor='mobilenetv2',
            quantize=False, lr=0.0005)
    else:
        train('../data/labels_ibug_300W_train_112_grey.npz', 
            '../data/unrot_train_pca.npz',
            '../data/checkpoints-custom/shapenet',
            image_size=112,
            in_channels=1,
            quantize=False, lr=0.001)
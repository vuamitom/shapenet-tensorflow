from shapnet import predict_landmarks
import numpy as np
import tensorflow as tf
import math

N_COMPONENTS = 12

class DataSet:
    def __init__(self, path, batch_size):
        ds = np.load(path)
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

def train(data_path, pca_path, save_path, checkpoint=None, lr=0.001):
    
    no_epoch = 1000
    batch_size = 32
    image_size = 224    
    components = get_pcomps(pca_path)

    # define input_variables
    inputs = tf.placeholder(tf.float32, shape=[None, image_size, image_size], name='input_images')
    labels = tf.placeholder(tf.float32, shape=[None, 68, 2], name='landmarks')

    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(no_epoch)

    iterator = dataset.make_initializable_iterator()
    next_batch, next_label = iterator.get_next()
    print('next batch ', next_batch)
    print('next label ', next_label)
    preds = predict_landmarks(next_batch, components)
    # define loss function
    l1_loss = tf.losses.absolute_difference(next_label, preds)
    mse_loss = tf.losses.mean_squared_error(next_label, preds)

    tf.summary.scalar('losses/l1_loss', l1_loss)
    tf.summary.scalar('losses/mse_loss', mse_loss)
    summary_op = tf.summary.merge_all()
    # define optimizer
    global_step = tf.train.get_or_create_global_step()
    optimizer = tf.train.AdamOptimizer(lr, 0.9, 0.999)
    train_op = optimizer.minimize(l1_loss, global_step)


    train_data, train_labels = None, None
    with np.load(data_path) as np_data:
        train_data = np_data['data']
        train_labels = np_data['labels']

    step = 0
    with tf.train.MonitoredTrainingSession(checkpoint_dir=save_path) as sess: 
        sess.run(iterator.initializer, feed_dict={
            inputs: train_data, labels: train_labels
        })
        while not sess.should_stop():
             _, summary, l1_loss_val, mse_loss_val = sess.run([train_op, summary_op, l1_loss, mse_loss])
             step += 1
             if step % 100 == 0:
                print('step ', step, 'l1 loss = ', l1_loss_val, 'mse_loss = ', mse_loss_val)


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
    train('../data/labels_ibug_300W_train.npz', 
        '../data/unrot_train_pca.npz',
        '../data/shapenet')
from shapnet import predict_landmarks
import numpy as np
import tensorflow as tf

N_COMPONENTS = 60

class DataSet:
    def __init__(self, path, batch_size):
        ds = np.load(path)
        self.data = ds['data']
        # normalize image data - first verify the value range
        # self.data = (self.data - 127.5) / 128.0
        self.labels = ds['labels']
        self.idx = 0
        self.batch_size = batch_size

    def reset_and_shuffle(self):
        pass

    def no_batches(self):
        return math.ceil(self.size() * 1.0/ self.batch_size)

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

def train(data_path, pca_path, lr=0.001):
    
    no_epoch = 1000
    batch_size = 20
    image_size = 224    
    components = get_pcomps(pca_path)

    # define input_variables
    inputs = tf.placeholder(tf.float32, shape=[batch_size, image_size, image_size], name='input_images')
    labels = tf.placeholder(tf.float32, shape=[batch_size, 68, 2], name='landmarks')
    preds = predict_landmarks(inputs, components)
    # define loss function
    l1_loss = tf.losses.absolute_difference(labels, preds)
    mse_loss = tf.losses.mean_squared_error(labels, preds)

    tf.summary.scalar('losses/l1_loss', l1_loss)
    tf.summary.scalar('losses/mse_loss', mse_loss)
    summary_op = tf.summary.merge_all()
    # define optimizer
    global_step = tf.train.get_or_create_global_step()
    optimizer = tf.train.AdamOptimizer(lr, 0.9, 0.999)
    train_op = optimizer.mimimize(l1_loss, global_step)

    
    # train_log_dir = ...
    # if not tf.gfile.Exists(log_dir):
    #     tf.gfile.MakeDirs(log_dir)

    # train_tensor = slim.learning.create_train_op(l1_loss, optimizer)
    # slim.learning.train(train_tensor, log_dir)
    ds = DataSet(data_path, batch_size)
    with tf.Session() as sess:        

        for epoch in range(0, no_epoch):
            # TODO: shuffle dataset 
            ds.reset_and_shuffle()
            # TODO: train single epoch
            for batch_no in range(0, ds.no_batches()):        
                batch_input, batch_label = ds.next_batch(batch_size)
                _, summary = sess.run([train_op, summary_op], feed_dict={input_images: batch_input, landmarks: batch_label})
            # end of epoch
            # eval
            # TODO: 
            print('end of epoch')

if __name__ == '__main__':
    train('../data/labels_ibug_300W_train.npz', '../data/unrot_train_pca.npz')
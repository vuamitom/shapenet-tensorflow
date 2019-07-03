# from shapnet import predict_landmarks as shapenet_predict_landmarks
from pfld import predict_landmarks as pfld_predict_landmarks, loss_fn as pfld_loss_fn
from pfld_custom import predict_landmarks as pfld_custom_predict_landmarks
import numpy as np
import tensorflow as tf
import math
import sys
import os

BATCH_SIZE = 256
NO_EPOCH = 1000
IMAGE_SIZE = 224

def normalize_landmarks(lmks, image_size, zero_mean = True):
    print('normalize_landmarks with zero_mean=',zero_mean)
    if zero_mean:
        return (lmks/image_size - 0.5) * 2
    else:
        return lmks/image_size
    # return lmks/image_size

def normalize_data(data):
    return (data - 0.5) * 2
    # return data

class DataSet:
    def __init__(self, path, batch_size, image_size, 
                            class_weight_path=None,
                            zero_mean=True):
        if class_weight_path is None:
            self.class_weights = None
        else:
            print('open class_weight_path', class_weight_path)
            with np.load(class_weight_path) as cw:
                self.class_weights = cw['weights']

        with np.load(path) as ds:
            # ds = np.load(path)
            self.data = ds['data']
            # normalize data
            self.data = normalize_data(self.data)# (self.data - 0.5) * 2
            self.labels = ds['labels']
            self.poses = ds['poses']
            self.labels = self.labels.reshape((-1, 1, 136)).squeeze()
            # if normalize_lmks:
            self.poses = self.poses / 90.0 # make it from -1 to 1 
            self.labels = normalize_landmarks(self.labels, image_size, zero_mean)

        if self.class_weights is None:
            self.class_weights = np.ones((self.labels.shape[0], 1))

        self.idx = 0
        self.batch_size = batch_size

    def reset_and_shuffle(self):
        permutation = np.random.permutation(self.labels.shape[0])
        self.data = self.data[permutation,:,:,:]
        self.labels = self.labels[permutation]
        self.poses = self.poses[permutation]
        self.class_weights = self.class_weights[permutation]
        self.idx = 0

    def reset(self):
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
        poses = self.poses[self.idx:n, :]
        class_weights = self.class_weights[self.idx:n, :]
        self.idx = n
        return data, labels, poses, class_weights


def train(data_path, save_path,
                            image_size=IMAGE_SIZE,
                            batch_size=BATCH_SIZE,
                            no_epoch=NO_EPOCH,
                            checkpoint=None, 
                            init_checkpoint=None,
                            quantize=True,
                            quant_delay=50000,
                            step_per_save=300,
                            eval_data_path=None,
                            class_weight_path=None,
                            depth_multiplier=1.0,
                            predict_fn=pfld_predict_landmarks,
                            lr=0.001,
                            **kwargs):

    print('train with image_size ', image_size, 
        ' quantize=', quantize, 
        'batch_size=', batch_size,         
        'learning rate = ', lr)

    inputs = tf.placeholder(tf.float32, shape=[None, image_size, image_size, 3], name='input_images')
    labels = tf.placeholder(tf.float32, shape=[None, 136], name='landmarks')
    poses = tf.placeholder(tf.float32, shape=[None, 3], name='poses')
    class_weights = tf.placeholder(tf.float32, shape=[None, 1], name='class_weights')

    preds, pose_preds, _ = predict_fn(inputs, image_size,
                                depth_multiplier=depth_multiplier,
                                is_training=True,
                                **kwargs)
    # define loss function
    loss = pfld_loss_fn(preds, pose_preds, labels, poses, class_weights)
    l1_loss = tf.losses.absolute_difference(labels*image_size, preds*image_size)
    aux_loss = tf.losses.absolute_difference(poses*90, pose_preds*90)

    if quantize:
        print('add custom op for quantize aware training after delay', quant_delay)
        tf.contrib.quantize.create_training_graph(input_graph=tf.get_default_graph(), quant_delay=quant_delay)
    global_step = tf.train.get_or_create_global_step() 
    print('adam optimizer with weigh decay 10^-6')   
    optimizer = tf.contrib.opt.AdamWOptimizer(10 ** -6, learning_rate=lr, beta1=0.9, name='Adam')

    # refer to https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/layers.py#L473
    update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        print('add dependency on "moving avg" for batch_norm')        
        train_op = optimizer.minimize(loss, global_step) 
    
    ds = DataSet(data_path, batch_size, image_size, class_weight_path=class_weight_path, zero_mean=True)
    print('Done loading dataset')

    eval_ds=None
    if eval_data_path is not None:
        eval_ds = DataSet(eval_data_path, 500, image_size)
        last_eval_res = None

    saver = tf.train.Saver()
    with tf.Session() as sess: 
        if checkpoint:
            print('restore from  checkpoint', checkpoint)
            saver.restore(sess, checkpoint)
        else:
            # if init_checkpoint:
            #     # a = tf.contrib.framework.get_variables_to_restore()
            #     # print('=====================',a)/
            #     tf.train.init_from_checkpoint(init_checkpoint, {
            #         # 'Backbone/expanded_conv/expand/weights/':'Backbone/expanded_conv/expand/weights',
            #         'aux_conv_4/': 'aux_conv_4',
            #         # 'aux_conv_1/': 'aux_conv_1',
            #         # 'aux_conv_2/': 'aux_conv_2',
            #         # 'aux_conv_3/': 'aux_conv_3',
            #         # 'aux_conv_4/': 'aux_conv_4',
            #         # 'aux_fc_1/': 'aux_fc_1',
            #         # 'aux_fc_2/': 'aux_fc_2'
            #         })
            # print('train from scratch')
            sess.run(tf.global_variables_initializer())

        for epoch in range(0, no_epoch):
            # sess.run(iterator.initializer, )
            # ds.reset_and_shuffle()
            ds.reset()
            for batch_no in range(0, ds.no_batches()):
                train_data, train_labels, train_poses, train_cw = ds.next_batch()        
                # train_data = (train_data - 0.5) * 2 
                _, loss_val, l1_loss_val, mse_loss_val, step = sess.run([train_op, loss, l1_loss, aux_loss, global_step], feed_dict={
                    inputs: train_data, labels: train_labels, poses: train_poses,
                    class_weights: train_cw
                })
                if step > 0 and step % 100 == 0:
                    print('step ', step, ' loss value = ', loss_val, 'l1 loss=', l1_loss_val, 'aux loss=', mse_loss_val)
                    if step % step_per_save == 0 and eval_ds is None:
                        # save
                        result_path = saver.save(sess, save_path, global_step=step)
                        print ('saved to ', result_path)

            print('end of epoch')
            if eval_ds is not None:
                eval_ds.reset_and_shuffle()
                eval_data, eval_labels, _ = eval_ds.next_batch()
                loss_val = sess.run(loss, feed_dict={inputs: eval_data, labels: eval_labels})
                print ('eval results: loss val =', loss_val)
                if last_eval_res is not None and loss_val < last_eval_res:
                    result_path = saver.save(sess, save_path, global_step=step)
                    print ('improved, saved to ', result_path)
                    last_eval_res = loss_val
                elif last_eval_res is None:
                    last_eval_res = loss_val


if __name__ == '__main__':
    train('../../data/labels_ibug_300W_train_80.npz', 
        '../../data/checkpoints-pfld-custom/pfld',
        batch_size=20,
        image_size=80,
        depth_multiplier=1.0,
        class_weight_path='../../data/labels_ibug_300W_train_80_classes.npz',
        predict_fn=pfld_custom_predict_landmarks,
        aux_start_layer='layer_7',       
        quantize=False, 
        mid_conv_n=4,
        lr=0.0001) 
    # else:
    #     train('../data/labels_ibug_300W_train_112_grey.npz', 
    #         '../data/unrot_train_pca.npz',
    #         '../data/checkpoints-112-1/shapenet',
    #         image_size=112,
    #         in_channels=1,
    #         extractor=extractors.custom_feature_extractor,
    #         quantize=False, lr=0.001)
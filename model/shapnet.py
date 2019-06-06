#coding:utf-8
import tensorflow as tf
from tensorflow.contrib import slim

def feature_extractor(inputs, in_channels, num_out_params):
    """ create feature extractor
    TODO: experiment with mobile net
    """
    norm_class = tf.contrib.layers.instance_norm
    p_dropout = False

    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        padding='VALID'):
        net = slim.stack(inputs, slim.conv2d, [(64, [7, 1]), (64, [1, 7])], scope='conv_1')
        net = slim.conv2d(net, 128, [7, 7], stride=2, scope='down_conv_1')
        if norm_class:
            net = norm_class(net, scope='norm_1')
        if p_dropout:
            net = slim.dropout(net, p_dropout, scope='dropout_1')

        net = slim.stack(net, slim.conv2d, [(128, [7, 1]), (128, [1, 7])], scope='conv_2')
        net = slim.conv2d(net, 256, [7, 7], stride=2, scope='down_conv_2')
        if norm_class:
            net = norm_class(net, scope='norm_2')
        if p_dropout:
            net = slim.dropout(net, p_dropout, scope='dropout_2')

        net = slim.stack(net, slim.conv2d, [(256, [5, 1]), (256, [1, 5])], scope='conv_3')
        net = slim.conv2d(net, 256, [5, 5], stride=2, scope='down_conv_3')
        if norm_class:
            net = norm_class(net, scope='norm_3')
        if p_dropout:
            net = slim.dropout(net, p_dropout, scope='dropout_3')

        net = slim.stack(net, slim.conv2d, [(256, [5, 1]), (256, [1, 5])], scope='conv_4')
        net = slim.conv2d(net, 128, [5, 5], stride=2, scope='down_conv_4')
        if norm_class:
            net = norm_class(net, scope='norm_4')
        if p_dropout:
            net = slim.dropout(net, p_dropout, scope='dropout_4')

        net = slim.stack(net, slim.conv2d, [(128, [3, 1]), (128, [1, 3]), (128, [3, 1]), (128, [1, 3])], scope='conv_5')
        net = slim.conv2d(net, num_out_params, [2, 2], scope='final_conv', activation_fn=None)

        return net


def get_shapenet(self, inputs):
    in_channels = 1
    num_out_params = 10
    feature_extractor = feature_extractor(inputs, in_channels, num_out_params)
    return feature_extractor
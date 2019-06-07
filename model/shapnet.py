#coding:utf-8
import tensorflow as tf
from tensorflow.contrib import slim

# N_COMPONENTS = 12
TRANSFORMS_OPS = dict(scale=1, rotation=1, translation=2)
N_DIMENS = 2

def shape_layer(shape_mean, components, features):
    """ shapes: eigen shape obtained by PCA
        features: features extracted from image """    
    batch_size = features.shape[0]
    expanded_components = tf.broadcast_to(components, [batch_size, *components.shape], name='expanded_components')
    features = tf.broadcast_to(features, expanded_components.shape)
    weighted_components = tf.multiply(expanded_components, features, name="weighted_components")

    expanded_means = tf.broadcast_to(shape_mean, [batch_size, *shape_mean.shape])
    shapes = tf.add(expanded_means, tf.reduce_sum(weighted_components, axis=1))
    return shapes

def transform_layer(shapes, transform_params):
    indices = dict()
    c = 0
    for k, v in TRANSFORMS_OPS:
        indices[k] = (c, c+v)
        c = c+v
    print('indices = ', indices)
    batch_size = shapes.shape[0]

    scale_params = transform_params[indices['scale'][0]: indices['scale'][1]]
    rotate_params = transform_params[indices['rotation'][0]: indices['rotation'][1]]
    translate_params = transform_params[indices['translation'][0]: indices['translation'][1]]

    # ensemble transfo
    trafo_matrix = tf.zeros([N_DIMENS+1, N_DIMENS+1], name='trafo_matrix')
    trafo_matrix[-1, -1] = 1
    trafo_matrix = tf.broadcast_to(trafo_matrix, [batch_size, *trafo_matrix.shape])
    trafo_matrix[:, 0, 0] = tf.multiply(scale_params, tf.cos(rotate_params))[:, 0]
    trafo_matrix[:, 0, 1] = tf.multiply(scale_params, tf.sin(rotate_params))[:, 0]
    trafo_matrix[:, 1, 0] = tf.multiply(-scale_params, tf.sin(rotate_params))[:, 0]
    trafo_matrix[:, 1, 1] = tf.multiply(scale_params, tf.cos(rotate_params))[:, 0]
    trafo_matrix[:, :-1, -1] = transform_params

    # concat 1 more dimen to shape
    shapes = tf.concat([shapes, tf.ones([*shapes.shape[:-1], 1])], -1)
    transformed_shapes = tf.matmul(shapes, tf.transpose(trafo_matrix, perm=[0, 2, 1]), name='transfo_matmul')
    return transformed_shapes[:, :, :-1]

def feature_extractor(inputs, num_out_params):
    """ create feature extractor
    TODO: experiment with mobile net
    """
    norm_class = tf.contrib.layers.instance_norm
    p_dropout = False
    # print(' inputs shape = ', inputs.shape)    
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


def predict_landmarks(inputs, pca_components):
    # shape means are stored at index 0
    shape_mean = tf.constant(pca_components[0], name='shape_means')
    components = tf.constant(pca_components[1:], name='components')

    in_channels = 1
    n_components = components.shape[0]

    n_transforms = 0
    for k, v in TRANSFORMS_OPS.items():
        n_transforms += v    

    num_out_params = n_components + n_transforms  
    print('num_out_params = ', n_components, n_transforms)

    if in_channels == 1:
        inputs = tf.expand_dims(inputs, -1)

    features = feature_extractor(inputs, num_out_params)
    features = tf.reshape(features, [inputs.shape[0], num_out_params, 1, 1])
    print('features shape ', features.shape, features[:, 0:n_components])
    shapes = shape_layer(shape_mean, components, features[:, 0:n_components])

    transformed_shapes = transform_layer(shapes, features[n_components:])
    return transformed_shapes
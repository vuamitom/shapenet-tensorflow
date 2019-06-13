#coding:utf-8
import tensorflow as tf
from tensorflow.contrib import slim
import extractors

# N_COMPONENTS = 12
TRANSFORMS_OPS = dict(scale=1, rotation=1, translation=2)
N_DIMENS = 2
FOR_TFLITE = True
FEATURE_EXTRACTOR = 'mobilenetv2'

def broadcast_to_batch(tensor, batch_size, name=None):
    if FOR_TFLITE:
        # print('tensor', tensor)
        multiples = tf.concat([[batch_size], tf.ones(tf.size(tf.shape(tensor)), dtype=tf.int32)], 0)
        # print('multiples = ', multiples)
        return tf.tile(tf.expand_dims(tensor, 0), multiples, name=name)
    else:
        return tf.broadcast_to(tensor, [batch_size, *tensor.shape], name=name)

def do_shape_transform(shapes, trafo_matrix, name=None):
    if FOR_TFLITE:
        # by right, we should do transpose here
        # but this cancel out the transpose needed
        no_lmks = tf.shape(shapes)[1]
        rows = []
        for r in range(0, 3): # this only works for tensor2 of shape (?, 3, 3)
            trafo = tf.tile(tf.expand_dims(trafo_matrix[:, r, :], 1), [1, no_lmks, 1])
            row = tf.reduce_sum(tf.multiply(shapes, trafo), -1, keepdims=True)
            rows.append(row)
        return tf.concat(rows, -1)            
    else:
        trafo_matrix = tf.transpose(trafo_matrix, perm=[0, 2, 1])
        return tf.matmul(shapes, trafo_matrix, name=name)

def do_cos(tensor, sin_tensor):
    if FOR_TFLITE:
        # calculate cos from sin 
        # since cos is not supported as of 1.13.1
        # cos = (1 - sin**2)** 0.5
        return tf.pow(1 - tf.pow(sin_tensor, 2), 0.5)
    else:
        return tf.cos(tensor)


def shape_layer(shape_mean, components, features):
    """ shapes: eigen shape obtained by PCA
        features: features extracted from image """    
    batch_size = tf.shape(features)[0]
    expanded_components = broadcast_to_batch(components, batch_size, name='expanded_components')
    # print('components tensor ', expanded_components)
    ec_shape = tf.shape(expanded_components)
    features =  tf.tile(features, [1, 1, ec_shape[2], ec_shape[3]])# tf.broadcast_to(features, tf.shape(expanded_components))
    # print('feature tensor ', features)
    weighted_components = tf.multiply(expanded_components, features, name="weighted_components")

    expanded_means = broadcast_to_batch(shape_mean, batch_size)# tf.broadcast_to(shape_mean, [batch_size, *shape_mean.shape])
    shapes = tf.add(expanded_means, tf.reduce_sum(weighted_components, axis=1))
    return shapes

def transform_layer(shapes, transform_params):
    indices = dict()
    c = 0
    for k, v in TRANSFORMS_OPS.items():
        indices[k] = (c, c+v)
        c = c+v
    # print('indices = ', indices)
    batch_size = tf.shape(shapes)[0]
    
    transform_params = tf.squeeze(transform_params, [2,3])# tf.reshape(transform_params, transform_params.shape[0:2])
    # print('transform_params ', transform_params)
    scale_params = transform_params[:, indices['scale'][0]: indices['scale'][1]]
    rotate_params = transform_params[:, indices['rotation'][0]: indices['rotation'][1]]
    translate_params = transform_params[:, indices['translation'][0]: indices['translation'][1]]

    # ensemble transfo
    # trafo_matrix = tf.zeros([N_DIMENS+1, N_DIMENS+1], name='trafo_matrix')
    # trafo_matrix[-1, -1] = 1
    # trafo_matrix = tf.broadcast_to(trafo_matrix, [batch_size, *trafo_matrix.shape])
    # trafo_matrix[:, 0, 0] = tf.multiply(scale_params, tf.cos(rotate_params))[:, 0]
    # trafo_matrix[:, 0, 1] = tf.multiply(scale_params, tf.sin(rotate_params))[:, 0]
    # trafo_matrix[:, 1, 0] = tf.multiply(-scale_params, tf.sin(rotate_params))[:, 0]
    # trafo_matrix[:, 1, 1] = tf.multiply(scale_params, tf.cos(rotate_params))[:, 0]
    # trafo_matrix[:, :-1, -1] = transform_params
    sin_rotate = tf.sin(rotate_params)
    cos_rotate = do_cos(rotate_params, sin_rotate)
    scale_x_cos = tf.multiply(scale_params, cos_rotate)
    scale_x_sin = tf.multiply(scale_params, sin_rotate)
    neg_scale_x_sin = tf.multiply(-scale_params, tf.sin(rotate_params))
    # print('scale_x_cos', scale_x_cos)
    fst_row = tf.concat([scale_x_cos, scale_x_sin], axis=-1)
    snd_row = tf.concat([neg_scale_x_sin, scale_x_cos], axis=-1)
    scale_rotate = tf.stack([fst_row, snd_row], axis=-1)
    temp = tf.concat([tf.shape(translate_params), [1]], 0)
    trafo_matrix = tf.concat([scale_rotate, tf.reshape(translate_params, temp)], axis=-1)
    last_row = broadcast_to_batch(tf.constant([[0, 0, 1]], dtype=tf.float32), batch_size)
    # print('last_row = ', last_row)
    trafo_matrix = tf.concat([trafo_matrix, last_row], axis=1)
    # print('trafo_matrix', trafo_matrix)
    # concat 1 more dimen to shape
    temp = tf.concat([tf.shape(shapes)[:-1], [1]], 0)
    shapes = tf.concat([shapes, tf.ones(temp)], -1)
    # print('shapes = ', shapes)
    transformed_shapes = do_shape_transform(shapes, trafo_matrix, name='transfo_matmul')
    # print('transformed_shapes', transformed_shapes)
    return transformed_shapes[:, :, :-1]



def predict_landmarks(inputs, pca_components, is_training=True, extractor=FEATURE_EXTRACTOR):
    print('using feature extractor ', extractor)
    # shape means are stored at index 0
    shape_mean = tf.constant(pca_components[0], name='shape_means', dtype=tf.float32)
    components = tf.constant(pca_components[1:], name='components', dtype=tf.float32)

    in_channels = 1 if len(inputs.shape) == 3 else 3 
    n_components = components.shape.as_list()[0]
    # print('shape ==== ', pca_components[1:].shape)
    n_transforms = 0
    for k, v in TRANSFORMS_OPS.items():
        n_transforms += v    

    num_out_params = n_components + n_transforms  
    # print('num_out_params = ', n_components, n_transforms)

    print('input channels ', in_channels)
    if in_channels == 1:
        inputs = tf.expand_dims(inputs, -1)

    if extractor == 'mobilenetv2':
        features = extractors.mobilenet_extract(inputs, num_out_params, is_training)
    else:
        features = extractors.custom_feature_extractor(inputs, num_out_params)
    print('feature after extractor ', features)
    features = tf.reshape(features, [-1, num_out_params, 1, 1])
    # print('features shape ', features.shape, features[:, 0:n_components])
    shapes = shape_layer(shape_mean, components, features[:, 0:n_components])

    transformed_shapes = transform_layer(shapes, features[:, n_components:])
    return transformed_shapes
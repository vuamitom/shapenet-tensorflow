import tensorflow as tf
from tensorflow.contrib import slim
# import sys
# sys.path.append('/home/tamvm/Projects/tensorflow-models/research/slim')
# sys.path.append('/home/tamvm/Projects/tensorflow-models/research')
# import nets.mobilenet.mobilenet_v2 as mobilenet_v2
# from nets.mobilenet import mobilenet
# from object_detection.utils import context_manager
# from object_detection.utils import ops
# from object_detection.models import feature_map_generators
# from object_detection.models.ssd_mobilenet_v2_feature_extractor import SSDMobileNetV2FeatureExtractor
# from object_detection.builders import hyperparams_builder
# from object_detection.protos import hyperparams_pb2
# from google.protobuf import text_format


# def mobilenet_extractor(inputs, num_outputs, is_training=True):
#     pad_to_multiple = 32
#     use_explicit_padding = True
#     depth_multiplier = 1.0
#     use_depthwise = True
#     override_base_feature_extractor_hyperparams = False
#     reuse_weights = None
#     min_depth = 16
#     specs = [
#             op(slim.conv2d, stride=2, num_outputs=64, kernel_size=[3, 3]),
#             # todo: Depthwise Conv3×3
#             op(slim.separable_conv2d, stride=1, kernel_size=[3, 3], num_outputs=None, multiplier_func=dummy_depth_multiplier),    
#             # 562×64Bottleneck 2 64 5 2
#             op(ops.expanded_conv, stride=2, num_outputs=64),            
#         ]
#     for _ in range(0, 4):
#         specs.append(op(ops.expanded_conv, stride=1, num_outputs=64))

#     # 282×64Bottleneck212812
#     specs.append(op(ops.expanded_conv, stride=2, num_outputs=128))

#     # 142×128Bottleneck412861    
#     for _ in range(0, 6):            
#         specs.append(op(ops.expanded_conv, 
#             expansion_size=expand_input(4, divisible_by=1), 
#             num_outputs=128,
#             stride=1))

#     specs.append(op(ops.expanded_conv, stride=1, num_outputs=16, scope='S1'))
#     specs.append(op(slim.conv2d, stride=2, kernel_size=[3, 3], num_outputs=32, scope='S2'))
#     specs.append(op(slim.conv2d, stride=1, kernel_size=[7, 7], num_outputs=128, scope='S3'))

#     # print('specs = ', specs, ' len = ', len(specs))

#     arch = dict(
#         defaults={
#             # Note: these parameters of batch norm affect the architecture
#             # that's why they are here and not in training_scope.
#             (slim.batch_norm,): {'center': True, 'scale': True},
#             (slim.conv2d, slim.fully_connected, slim.separable_conv2d): {
#                 'normalizer_fn': slim.batch_norm, 'activation_fn': tf.nn.relu6
#             },
#             (ops.expanded_conv,): {
#                 'expansion_size': expand_input(2),
#                 'split_expansion': 1,
#                 'normalizer_fn': slim.batch_norm,
#                 'residual': True
#             },
#             (slim.conv2d, slim.separable_conv2d): {'padding': 'SAME'}
#         },

#         spec=specs
#     )


#     with tf.variable_scope('Backbone', reuse=reuse_weights) as scope:
#         with slim.arg_scope(
#             mobilenet_v2.training_scope(is_training=is_training, bn_decay=0.9997)), \
#             slim.arg_scope(
#               [mobilenet.depth_multiplier], min_depth=min_depth):
#             with (slim.arg_scope(conv_hyperparams_fn(is_training=is_training))
#                 if override_base_feature_extractor_hyperparams else
#                 context_manager.IdentityContextManager()):
#                 _, image_features = mobilenet_v2.mobilenet_base(
#                   od_ops.pad_to_multiple(inputs, pad_to_multiple),                  
#                   depth_multiplier=depth_multiplier,
#                   is_training=is_training,
#                   use_explicit_padding=use_explicit_padding,
#                   conv_defs=arch,
#                   scope=scope)

#                 S1 = image_features['layer_15']
#                 S2 = image_features['layer_16']
#                 S3 = image_features['layer_17']
#                 # batch_size = tf.shape(S1)[0]
#                 S1 = slim.flatten(S1, scope='S1flatten') # tf.reshape(S1, [batch_size, -1])
#                 S2 = slim.flatten(S2, scope='S2flatten') # [batch_size, -1])
#                 S3 = slim.flatten(S3, scope='S3flatten') # [batch_size, -1])
#                 before_dense = tf.concat([S1, S2, S3], 1)
#                 # print('before dense = ', before_dense)
#                 # before_dense.set_shape([None, 100 ])
#                 return image_features, slim.fully_connected(before_dense, num_outputs)

# def mobilenet_extract(preprocessed_inputs, num_outputs, is_training=True):
#     feature_map_layout = {
#         'from_layer': ['layer_15/expansion_output', 'layer_19', '', '', '', '', '', ''],
#         'layer_depth': [-1, -1, 512, 256, 128, 64, 32, num_outputs]
#     }
#     return _mobilenet_extractor(preprocessed_inputs, num_outputs, feature_map_layout, is_training=is_training)

# def mobilenet_extract_v2(preprocessed_inputs, num_outputs, is_training=True):
#     feature_map_layout = {
#         'from_layer': ['layer_15/expansion_output', 'layer_19', '', '', '', '', '', '', ''],
#         'layer_depth': [-1, -1, 512, 256, 256, 128, 64, 32, num_outputs]
#     }
#     return _mobilenet_extractor(preprocessed_inputs, num_outputs, feature_map_layout, is_training=is_training)

# def mobilenet_extract_v4(preprocessed_inputs, num_outputs, is_training=True):
#     feature_map_layout = {
#         'from_layer': ['layer_15/expansion_output', 'layer_19', '', '', '', '', '', '', ''],
#         'layer_depth': [-1, -1, 512, 256, 256, 128, 128, 128, num_outputs]
#     }
#     return _mobilenet_extractor(preprocessed_inputs, num_outputs, feature_map_layout, is_training=is_training)

# def mobilenet_extract_v5(preprocessed_inputs, num_outputs, is_training=True):
#     feature_map_layout = {
#         'from_layer': ['layer_15/expansion_output', 'layer_19', '', '', '', '', '', '', ''],
#         'layer_depth': [-1, -1, 512, 512, 256, 256, 128, 128, num_outputs]
#     }
#     return _mobilenet_extractor(preprocessed_inputs, num_outputs, feature_map_layout, is_training=is_training)

# def mobilenet_extract_v3(preprocessed_inputs, num_outputs, is_training=True):
#     feature_map_layout = {
#         'from_layer': ['layer_15/expansion_output', 'layer_19', '', '', '', '', '', ''],
#         'layer_depth': [-1, -1, 512, 256, 256, 128, 64, 32]
#     }
#     features = _mobilenet_extractor(preprocessed_inputs, num_outputs, feature_map_layout, is_training=is_training)
#     return slim.fully_connected(features, num_outputs)


def depthwise_conv_feature_extractor(inputs, num_out_params, is_training=True):
    return original_paper_feature_extractor(inputs, num_out_params, is_training=is_training, use_depthwise=True)
    
def original_paper_feature_extractor(inputs, num_out_params, is_training=True, use_depthwise=False):
    """ create feature extractor for 224x224 input
    """
    norm_class = tf.contrib.layers.instance_norm
    conv2d_class = slim.separable_conv2d if use_depthwise else slim.conv2d
    # conv2d_class = slim.conv2d
    p_dropout = False
    # print(' inputs shape = ', inputs.shape)    
    with slim.arg_scope([conv2d_class],
                        activation_fn=tf.nn.relu,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        padding='VALID'):
        net = slim.stack(inputs, conv2d_class, [(64, [7, 1]), (64, [1, 7])], scope='conv_1')
        net = conv2d_class(net, 128, [7, 7], stride=2, scope='down_conv_1')
        if norm_class:
            net = norm_class(net, scope='norm_1')
        if p_dropout:
            net = slim.dropout(net, p_dropout, scope='dropout_1')

        net = slim.stack(net, conv2d_class, [(128, [7, 1]), (128, [1, 7])], scope='conv_2')
        net = conv2d_class(net, 256, [7, 7], stride=2, scope='down_conv_2')
        if norm_class:
            net = norm_class(net, scope='norm_2')
        if p_dropout:
            net = slim.dropout(net, p_dropout, scope='dropout_2')

        net = slim.stack(net, conv2d_class, [(256, [5, 1]), (256, [1, 5])], scope='conv_3')
        net = conv2d_class(net, 256, [5, 5], stride=2, scope='down_conv_3')
        if norm_class:
            net = norm_class(net, scope='norm_3')
        if p_dropout:
            net = slim.dropout(net, p_dropout, scope='dropout_3')

        net = slim.stack(net, conv2d_class, [(256, [5, 1]), (256, [1, 5])], scope='conv_4')
        net = conv2d_class(net, 128, [5, 5], stride=2, scope='down_conv_4')
        if norm_class:
            net = norm_class(net, scope='norm_4')
        if p_dropout:
            net = slim.dropout(net, p_dropout, scope='dropout_4')

        net = slim.stack(net, conv2d_class, [(128, [3, 1]), (128, [1, 3]), (128, [3, 1]), (128, [1, 3])], scope='conv_5')
        net = conv2d_class(net, num_out_params, [2, 2], scope='final_conv', activation_fn=None)
        return net
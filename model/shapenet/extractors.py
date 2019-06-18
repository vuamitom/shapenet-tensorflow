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

# import nets.mobilenet.conv_blocks as conv_blocks

# def conv_hyperparams_fn2(add_batch_norm=True, is_training=True):
#     conv_hyperparams = hyperparams_pb2.Hyperparams()
#     conv_hyperparams_text_proto = """
#       activation: RELU_6
#       regularizer {
#         l2_regularizer {
#             weight: 4.0e-05
#         }
#       }
#       initializer {
#         truncated_normal_initializer {
#             mean: 0.0
#             stddev: 0.03
#         }
#       }
#     """
#     if add_batch_norm:
#       batch_norm_proto = """
#         batch_norm {
#           decay: 0.9997
#           center: true
#           scale: true
#           epsilon: 0.001
#           train: true
#         }
#       """

#     conv_hyperparams_text_proto += batch_norm_proto
#     text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams)

#     scope_fn = hyperparams_builder.build(conv_hyperparams, True)
#     return scope_fn()

# def conv_hyperparams_fn(**kwargs):
#     with tf.contrib.slim.arg_scope([]) as sc:
#       return sc

# def _mobilenet_extractor(preprocessed_inputs, num_outputs, feature_layout, is_training=True, conv_defs=None):
#     pad_to_multiple = 32
#     use_explicit_padding = True
#     depth_multiplier = 1.0
#     use_depthwise = True
#     override_base_feature_extractor_hyperparams = False
#     reuse_weights = None
#     min_depth = 16

#     if not conv_defs:
#         conv_defs = mobilenet_v2.V2_DEF

#     feature_map_layout = {
#         'use_depthwise': use_depthwise,
#         'use_explicit_padding': use_explicit_padding,
#     }
#     feature_map_layout.update(feature_layout)

#     with tf.variable_scope('MobilenetV2', reuse=reuse_weights) as scope:
#         with slim.arg_scope(
#             mobilenet_v2.training_scope(is_training=None, bn_decay=0.9997)), \
#             slim.arg_scope(
#               [mobilenet.depth_multiplier], min_depth=min_depth):
#             with (slim.arg_scope(conv_hyperparams_fn(is_training=is_training))
#                 if override_base_feature_extractor_hyperparams else
#                 context_manager.IdentityContextManager()):
#                 _, image_features = mobilenet_v2.mobilenet_base(
#                   ops.pad_to_multiple(preprocessed_inputs, pad_to_multiple),
#                   final_endpoint='layer_19',
#                   depth_multiplier=depth_multiplier,
#                   use_explicit_padding=use_explicit_padding,
#                   conv_defs=conv_defs,
#                   scope=scope)
#             with slim.arg_scope(conv_hyperparams_fn(is_training=is_training)):
#               feature_maps = feature_map_generators.multi_resolution_feature_maps(
#                   feature_map_layout=feature_map_layout,
#                   depth_multiplier=depth_multiplier,
#                   min_depth=min_depth,
#                   insert_1x1_conv=True,
#                   image_features=image_features)

#     print ('keys = ', feature_maps.keys())
#     return feature_maps[list(feature_maps.keys())[-1]]

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
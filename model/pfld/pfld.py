import tensorflow as tf 
from tensorflow.contrib import slim
import sys
sys.path.append('/home/tamvm/Projects/tensorflow-models/research/slim')
import nets.mobilenet.mobilenet_v2 as mobilenet_v2
from nets.mobilenet import mobilenet as lib
from nets.mobilenet import conv_blocks as ops
from object_detection.utils import ops as od_ops
from nets.mobilenet import mobilenet
from object_detection.utils import context_manager
expand_input = ops.expand_input_by_factor
op = lib.op
# https://arxiv.org/pdf/1902.10859.pdf

def dummy_depth_multiplier(output_params,
                     multiplier,
                     divisible_by=8,
                     min_depth=8,
**unused_kwargs):
    return

def backbone_net(inputs):
    pad_to_multiple = 32
    use_explicit_padding = True
    depth_multiplier = 1.0
    use_depthwise = True
    override_base_feature_extractor_hyperparams = False
    reuse_weights = None
    min_depth = 16
    specs = [
            op(slim.conv2d, stride=2, num_outputs=64, kernel_size=[3, 3]),
            # todo: Depthwise Conv3×3
            op(slim.separable_conv2d, stride=1, kernel_size=[3, 3], num_outputs=None, multiplier_func=dummy_depth_multiplier),    
            # 562×64Bottleneck 2 64 5 2
            op(ops.expanded_conv, stride=2, num_outputs=64),            
        ]
    for _ in range(0, 4):
        specs.append(op(ops.expanded_conv, stride=1, num_outputs=64))

    # 282×64Bottleneck212812
    specs.append(op(ops.expanded_conv, stride=2, num_outputs=128))

    # 142×128Bottleneck412861    
    for _ in range(0, 6):            
        specs.append(op(ops.expanded_conv, 
            expansion_size=expand_input(4, divisible_by=1), 
            num_outputs=128,
            stride=1))

    specs.append(op(ops.expanded_conv, stride=1, num_outputs=16, scope='S1'))
    specs.append(op(slim.conv2d, stride=2, kernel_size=[3, 3], num_outputs=32, scope='S2'))
    specs.append(op(slim.conv2d, stride=1, kernel_size=[7, 7], num_outputs=128, scope='S3'))

    print('specs = ', specs, ' len = ', len(specs))

    arch = dict(
        defaults={
            # Note: these parameters of batch norm affect the architecture
            # that's why they are here and not in training_scope.
            (slim.batch_norm,): {'center': True, 'scale': True},
            (slim.conv2d, slim.fully_connected, slim.separable_conv2d): {
                'normalizer_fn': slim.batch_norm, 'activation_fn': tf.nn.relu6
            },
            (ops.expanded_conv,): {
                'expansion_size': expand_input(2),
                'split_expansion': 1,
                'normalizer_fn': slim.batch_norm,
                'residual': True
            },
            (slim.conv2d, slim.separable_conv2d): {'padding': 'SAME'}
        },

        spec=specs
    )


    with tf.variable_scope('Backbone', reuse=reuse_weights) as scope:
        with slim.arg_scope(
            mobilenet_v2.training_scope(is_training=None, bn_decay=0.9997)), \
            slim.arg_scope(
              [mobilenet.depth_multiplier], min_depth=min_depth):
            with (slim.arg_scope(conv_hyperparams_fn(is_training=is_training))
                if override_base_feature_extractor_hyperparams else
                context_manager.IdentityContextManager()):
                _, image_features = mobilenet_v2.mobilenet_base(
                  od_ops.pad_to_multiple(inputs, pad_to_multiple),                  
                  depth_multiplier=depth_multiplier,
                  use_explicit_padding=use_explicit_padding,
                  conv_defs=arch,
                  scope=scope)
                # do a fully connected layer here
                # TODO
                print('image features', image_features)
                S1 = image_features['layer_15/output']
                S2 = image_features['layer_16']
                S3 = image_features['layer_17']
                # batch_size = tf.shape(S1)[0]
                S1 = slim.flatten(S1, scope='S1flatten') # tf.reshape(S1, [batch_size, -1])
                S2 = slim.flatten(S2, scope='S2flatten') # [batch_size, -1])
                S3 = slim.flatten(S3, scope='S3flatten') # [batch_size, -1])
                before_dense = tf.concat([S1, S2, S3], 1)
                print('before dense = ', before_dense)
                # before_dense.set_shape([None, 100 ])
                return slim.fully_connected(before_dense, 136)


def predict_landmarks(inputs, *args, **kwargs):
    output = backbone_net(inputs)
    print('output = ', output)
    return output

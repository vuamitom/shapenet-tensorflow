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
import math
# https://arxiv.org/pdf/1902.10859.pdf

def conv_hyperparams_fn(**kwargs):
    with tf.contrib.slim.arg_scope([]) as sc:
        return sc

def auxiliary_net(inputs, image_size, is_training=True):
    # TODO: handle is_training for slim.batch_norm
    # print('----------inputs to auxiliary_net--------------', inputs)
    nodes = []
    with slim.arg_scope([slim.conv2d],
                    weights_initializer=slim.xavier_initializer(),
                    normalizer_fn=slim.batch_norm,
                    activation_fn=tf.nn.relu6):
        with slim.arg_scope([slim.batch_norm], is_training=is_training, center=True, scale=True):
            # net = slim.stack(inputs, conv2d_class, [(64, [7, 1]), (64, [1, 7])], scope='conv_1')
            net = slim.conv2d(inputs, 128, [3, 3], stride=2, scope='aux_conv_1', activation_fn=tf.nn.elu)
            # print('after first conv2d', net)
            nodes.append(net)
            net = slim.conv2d(net, 128, [3, 3], stride=1, scope='aux_conv_2')
            nodes.append(net)
            # print('after second conv2d', net)
            net = slim.conv2d(net, 32, [3, 3], stride=2, scope='aux_conv_3')
            nodes.append(net)
            # print('after second to last conv2d', net)
            kernel_size = ([7, 7] if image_size==112 else ([5,5] if image_size == 80 else [4, 4]))
            net = slim.conv2d(net, 128, kernel_size, stride=1, scope='aux_conv_4', 
                                activation_fn=tf.nn.elu,
                                padding='VALID')
            nodes.append(net)
            # print('after last conv2d', net)
            net = slim.flatten(net)
            with slim.arg_scope([slim.fully_connected], normalizer_fn=None, activation_fn=tf.nn.tanh):
                pre_roll = slim.fully_connected(net, 16, scope='pre_roll')
                nodes.append(pre_roll)
                # print('after first full connected', net)
                roll = slim.fully_connected(pre_roll, 1, scope='roll')
                nodes.append(roll)

                pre_pitch = slim.fully_connected(net, 16, scope='pre_pitch')
                nodes.append(pre_pitch)
                # print('after first full connected', net)
                pitch = slim.fully_connected(pre_pitch, 1, scope='pitch')
                nodes.append(pitch)

                pre_yaw = slim.fully_connected(net, 16, scope='pre_yaw')
                nodes.append(pre_yaw)
                # print('after first full connected', net)
                yaw = slim.fully_connected(pre_yaw, 1, scope='yaw')
                nodes.append(yaw)

                poses = tf.concat([roll, pitch, yaw], -1)
                nodes.append(poses)
            # print('after last full connected', net)
            print('------ auxiliary_net-------')
            for i in range(len(nodes)):
                print('layer_' + str(i+1), nodes[i])
            # net = tf.reshape(net, [-1, 3])
            # print('auxiliary_net output', poses)
        return poses, nodes        
        
def backbone_net(inputs, image_size, is_training=True, depth_multiplier=0.5):
    
    pad_to_multiple = 10
    use_explicit_padding = False
    depth_multiplier = depth_multiplier

    print('construct backbone_net for image_size', image_size, 'depth_multiplier = ', depth_multiplier)
    use_depthwise = True
    override_base_feature_extractor_hyperparams = False
    reuse_weights = None
    min_depth = 16

    specs = [
            op(slim.conv2d, stride=2, num_outputs=64, kernel_size=[3, 3], activation_fn=tf.nn.elu),
            # todo: Depthwise Conv3×3
            op(ops.expanded_conv, stride=1, kernel_size=[3, 3], num_outputs=64),    
            # 562×64Bottleneck 2 64 5 2
            op(slim.max_pool2d, kernel_size=[3, 3], padding='SAME', stride=1),

            op(ops.expanded_conv, stride=2, num_outputs=64),            
        ]
    for _ in range(0, 4):
        specs.append(op(ops.expanded_conv, stride=1, num_outputs=64))

    # 282×64Bottleneck212812
    specs.append(op(ops.expanded_conv, stride=2, num_outputs=128))

    # 142×128Bottleneck412861    
    for _ in range(0, 4):            
        specs.append(op(ops.expanded_conv, 
            expansion_size=expand_input(4), 
            num_outputs=128,
            stride=1))

    kernel_size = [5,5]
    specs.append(op(ops.expanded_conv, stride=1, num_outputs=16, scope='S1'))
    specs.append(op(slim.conv2d, stride=2, kernel_size=[3, 3], 
                        num_outputs=32, scope='S2', activation_fn=tf.nn.elu))
    specs.append(op(slim.conv2d, stride=1, kernel_size=kernel_size, 
                        num_outputs=128, scope='S3', padding='VALID', activation_fn=tf.nn.elu))

    # print('specs = ', specs, ' len = ', len(specs))

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
                'residual': True,
            },
            (slim.conv2d, slim.separable_conv2d): {'padding': 'SAME', 'weights_initializer': slim.xavier_initializer()}
        },

        spec=specs
    )

    print('input to backbone_net ' , inputs)
    with tf.variable_scope('Backbone', reuse=reuse_weights) as scope:
        with slim.arg_scope(
            mobilenet_v2.training_scope(is_training=is_training, bn_decay=0.9997)), \
            slim.arg_scope(
              [mobilenet.depth_multiplier], min_depth=min_depth):
            with (slim.arg_scope(conv_hyperparams_fn(is_training=is_training))
                if override_base_feature_extractor_hyperparams else
                context_manager.IdentityContextManager()):
                _, image_features = mobilenet_v2.mobilenet_base(
                  od_ops.pad_to_multiple(inputs, pad_to_multiple),                  
                  depth_multiplier=depth_multiplier,
                  is_training=is_training,
                  use_explicit_padding=use_explicit_padding,
                  conv_defs=arch,
                  scope=scope)
                # do a fully connected layer here
                # TODO

                print('image image_features', image_features.keys())

                layer_15 = image_features['layer_14']
                layer_16 = image_features['layer_15']
                layer_17 = image_features['layer_16']
                # batch_size = tf.shape(S1)[0]                

                S1 = slim.flatten(layer_15, scope='S1flatten') # tf.reshape(S1, [batch_size, -1])
                S2 = slim.flatten(layer_16, scope='S2flatten') # [batch_size, -1])
                S3 = slim.flatten(layer_17, scope='S3flatten') # [batch_size, -1])
                before_dense = tf.concat([S1, S2, S3], 1)
                
                for i in range(1, 17):
                    print('layer_' + str(i), image_features['layer_' + str(i)])
                # print('layer_17', layer_17)
                print('S1', S1)
                print('S2', S2)
                print('S3', S3)

                # to_test = slim.conv2d(image_features['layer_19'])
                print('before fully_connected', before_dense)
                with slim.arg_scope([slim.fully_connected], 
                                        weights_initializer=slim.xavier_initializer(),
                                        normalizer_fn=None, 
                                        activation_fn=tf.nn.tanh):
                    pre_chin = slim.fully_connected(before_dense, 68)
                    pre_left_eye_brow = slim.fully_connected(before_dense, 20)
                    pre_right_eye_brow = slim.fully_connected(before_dense, 20)
                    pre_nose = slim.fully_connected(before_dense, 36)
                    pre_left_eye = slim.fully_connected(before_dense, 24)
                    pre_right_eye = slim.fully_connected(before_dense, 24)
                    pre_mouth = slim.fully_connected(before_dense, 80)

                    chin = slim.fully_connected(pre_chin, 34)
                    left_eye_brow = slim.fully_connected(pre_left_eye_brow, 10)
                    right_eye_brow = slim.fully_connected(pre_right_eye_brow, 10)
                    nose = slim.fully_connected(pre_nose, 18)
                    left_eye = slim.fully_connected(pre_left_eye, 12)
                    right_eye = slim.fully_connected(pre_right_eye, 12)
                    mouth = slim.fully_connected(pre_mouth, 40)

                    landmarks = tf.concat([chin, left_eye_brow, right_eye_brow, nose, left_eye, right_eye, mouth], -1)
                    return image_features, landmarks, None


def predict_landmarks(inputs, image_size, is_training=True, depth_multiplier=0.5, *args, **kwargs):
    mobilenet_output, landmarks, unused = backbone_net(inputs, image_size, is_training=is_training, depth_multiplier=depth_multiplier)
    # print('output = ', output)
    # print('layer 0', mobilenet_output['layer_0'])
    # print('layer 4/output ', mobilenet_output['layer_4/output'])
    # print('layer 15', mobilenet_output['layer_15'])
    # print('layer 15/output ', mobilenet_output['layer_15/output'])
    if is_training:
        pose, _ = auxiliary_net(mobilenet_output['layer_5'], image_size)
    else:
        pose = None
    return landmarks, pose, unused

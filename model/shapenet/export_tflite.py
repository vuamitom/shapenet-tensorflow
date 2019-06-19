from shapnet import predict_landmarks
from train import get_pcomps
import tensorflow as tf
import os
import extractors
import numpy as np

def representative_dataset():
    samples = None
    with np.load('../../data/labels_ibug_300W_train_224_grey.npz') as ds:
        samples = ds['data'][0:500]
    for input_value in samples:
        yield [input_value.reshape((1, *input_value.shape))]

def export(output_path, pca_path, model_path,
            quantize_uint8=True,
            image_size=224,
            feature_extractor=extractors.original_paper_feature_extractor,
            in_channels=1):

    components = get_pcomps(pca_path)
    input_shape = [1, image_size, image_size]
    inputs = tf.placeholder(tf.float32, shape=input_shape, name='input_images')
    preds = predict_landmarks(inputs, components, 
        is_training=False, 
        feature_extractor=feature_extractor)

    inputs = [inputs]
    outputs = [preds]
    # if quantize:
    #     g = tf.get_default_graph()
    #     tf.contrib.quantize.create_eval_graph(input_graph=g)
    with tf.Session() as sess:            
        
        # sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        converter = tf.lite.TFLiteConverter.from_session(sess, inputs, outputs)
        # converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
        if not quantize_uint8:
            # converter.inference_type = tf.uint8
            # converter.quantized_input_stats['input_images'] = (128.0, 128) # (mean, std)
            # converter.default_ranges_stats = (0, 1)
            converter.allow_custom_ops = False    
            converter.post_training_quantize = True
        else:
            converter.allow_custom_ops = False
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_dataset

        
        tflite_model = converter.convert()
        # op = os.path.join(output_dir,  'shapenet.tflite')
        with open(output_path, 'wb') as f:
            f.write(tflite_model)

if __name__ == '__main__':
    pca_path = '../../data/unrot_train_pca.npz'
    use_depthwise = True 
    quantize_uint8 = False
    if use_depthwise:
        output_path = '../../data/shapenet-depthwise-quant.tflite'
        model_path = '../../data/checkpoints-depthwise/shapenet-20500'
        feature_extractor = extractors.depthwise_conv_feature_extractor
    else:
        output_path = '../../data/shapenet.tflite'    
        model_path = '../../data/checkpoints/shapenet-89000'
        feature_extractor = extractors.original_paper_feature_extractor
    
    
    export(output_path, pca_path, model_path, 
        feature_extractor=feature_extractor,
        quantize_uint8=quantize_uint8)


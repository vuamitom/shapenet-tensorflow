from pfld import predict_landmarks
import tensorflow as tf
import os
import numpy as np 
from train_pfld import normalize_data

def get_representative_dataset(image_size):
    if image_size == 64:
        data_path = '../../data/labels_ibug_300W_train_64.npz'
    elif image_size == 80:
        data_path = '../../data/labels_ibug_300W_train_80.npz'   

    def representative_dataset():
        samples = None
        with np.load(data_path) as ds:
            samples = ds['data'][0:500]
        samples = normalize_data(samples)
        for input_value in samples:
            # print('input value shape ', input_value.shape)
            yield [input_value.reshape((1, *input_value.shape))]
    return representative_dataset

def export(output_path, model_path,            
            image_size=112,
            quantize_uint8=False,
            depth_multiplier=1.0,
            in_channels=1):

    input_shape = [1, image_size, image_size, 3]
    inputs = tf.placeholder(tf.float32, shape=input_shape, name='input_images')
    preds,_,_ = predict_landmarks(inputs, image_size,
        is_training=False,
        depth_multiplier=depth_multiplier)
    # print('nodes name ', [n.name for n in tf.get_default_graph().as_graph_def().node])

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
        # if quantize:
        if not quantize_uint8:            
            # converter.inference_type = tf.uint8            
            # converter.quantized_input_stats = {}
            # converter.quantized_input_stats['input_images'] = (128.0, 128) # (mean, std)
            # converter.default_ranges_stats = (0, 255)
            print('do post quantize float')
            converter.allow_custom_ops = False
            converter.post_training_quantize = True
        else:
            print('do post quantize int')
            converter.allow_custom_ops = False
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = get_representative_dataset(image_size)

        tflite_model = converter.convert()
        # op = os.path.join(output_dir,  'shapenet.tflite')
        with open(output_path, 'wb') as f:
            f.write(tflite_model)

if __name__ == '__main__':
    output_path = '../../data/pfld-64-075m-quant.tflite'
    model_path = '../../data/checkpoints-pfld-64-075m/pfld-73200'
    export(output_path, model_path, image_size=64, quantize_uint8=True, depth_multiplier=0.75)


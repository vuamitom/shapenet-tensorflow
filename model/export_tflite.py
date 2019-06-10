import sys
sys.path.append('../preprocess')
from prepare_data import IMAGE_SIZE
from shapnet import predict_landmarks
from train import get_pcomps
import tensorflow as tf
import os

def export(output_dir, pca_path, model_path):
    components = get_pcomps(pca_path)
    quantize_aware_training = False


    inputs = tf.placeholder(tf.float32, shape=[1, IMAGE_SIZE, IMAGE_SIZE], name='input_images')
    preds = predict_landmarks(inputs, components)
    inputs = [inputs]
    outputs = [preds]

    with tf.Session() as sess:            
        if quantize_aware_training:
            g = tf.get_default_graph()
            tf.contrib.quantize.create_eval_graph(input_graph=g)

        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        converter = tf.lite.TFLiteConverter.from_session(sess, inputs, outputs)
        # converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
        # if quantize_aware_training:
        # converter.inference_type = tf.uint8
        converter.allow_custom_ops = False
        # converter.quantized_input_stats = {}
        # converter.quantized_input_stats['input_images'] = (127.5, 128) # (mean, std)
            # converter.default_ranges_min = 0
            # converter.default_ranges_max = 128
        converter.post_training_quantize = True

        tflite_model = converter.convert()
        op = os.path.join(output_dir,  'shapenet.tflite')
        with open(op, 'wb') as f:
            f.write(tflite_model)

if __name__ == '__main__':
    output_dir = '../data/'
    model_path = '../data/checkpoints/shapenet-64200'
    pca_path = '../data/unrot_train_pca.npz'
    export(output_dir, pca_path, model_path)


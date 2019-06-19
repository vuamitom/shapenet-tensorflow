import sys
sys.path.append('../preprocess')
from prepare_data import IMAGE_SIZE
from shapnet import predict_landmarks as shapenet_predict_landmarks
from train import get_pcomps
import tensorflow as tf
import os
import extractors

def export(output_dir, pca_path, model_path,
            quantize=True,
            image_size=IMAGE_SIZE,
            in_channels=1):
    components = get_pcomps(pca_path)
    input_shape = [1, image_size, image_size] if in_channels == 1 else [1, image_size, image_size, 3]
    inputs = tf.placeholder(tf.float32, shape=input_shape, name='input_images')
    preds = predict_landmarks(inputs, components, 
        is_training=False, 
        feature_extractor=extractors.custom_feature_extractor)
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
        #     converter.inference_type = tf.uint8
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


def export_pfld():
    pass

if __name__ == '__main__':
    output_dir = '../data/'
    model_path = '../data/checkpoints/shapenet-89000'
    pca_path = '../data/unrot_train_pca.npz'
    export(output_dir, pca_path, model_path, in_channels=1, image_size=224)


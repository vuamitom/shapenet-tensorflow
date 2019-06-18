from pfld import predict_landmarks
import tensorflow as tf
import os

def export(output_path, model_path,
            quantize=True,
            image_size=112,
            quantize_uint8=False,
            in_channels=1):

    input_shape = [1, image_size, image_size, 3]
    inputs = tf.placeholder(tf.float32, shape=input_shape, name='input_images')
    preds = predict_landmarks(inputs, 
        is_training=False)

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
        if quantize_uint8:
            converter.inference_type = tf.uint8
            converter.allow_custom_ops = False
            converter.quantized_input_stats = {}
            converter.quantized_input_stats['input_images'] = (127.0, 128) # (mean, std)
            # converter.default_ranges_stats = (-1, 1)
        converter.post_training_quantize = True
        tflite_model = converter.convert()
        # op = os.path.join(output_dir,  'shapenet.tflite')
        with open(output_path, 'wb') as f:
            f.write(tflite_model)

if __name__ == '__main__':
    output_path = '../../data/pfld-64.tflite'
    model_path = '../../data/checkpoints-pfld-64/pfld-218400'
    export(output_path, model_path, image_size=64, quantize_uint8=True)


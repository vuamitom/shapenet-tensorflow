import sys
sys.path.append('../../preprocess')

import numpy as np
import tensorflow as tf
from pfld import predict_landmarks as pfld_predict_landmarks
from pfld_custom import predict_landmarks as pfld_custom_predict_landmarks
from skimage.color import rgb2gray
import cv2
import dlib
from skimage.transform import resize
from prepare_data import IMAGE_SIZE, view_img
from skimage.transform import resize
import matplotlib
from train_pfld import normalize_data
matplotlib.use("TkAgg")
# IMAGE_SIZE = 224

def predict(data, model_path, predict_fn, image_size=IMAGE_SIZE, depth_multiplier=1.0, **kwargs):
    input_shape = [None, image_size, image_size, 3]
    inputs = tf.placeholder(tf.float32, shape=input_shape, name='input_images')
    preds, _, _ = predict_fn(inputs, image_size, is_training=False, depth_multiplier=depth_multiplier, **kwargs)    
    print('predict tensor = ', preds)
    saver = tf.train.Saver()
    # g = tf.get_default_graph()
    # tf.contrib.quantize.create_eval_graph(input_graph=g)
    with tf.Session() as sess:         
        saver.restore(sess, model_path)   
        # sess.run(tf.global_variables_initializer())
        results = sess.run(preds, feed_dict={inputs: data})
        print('landmarks = ', results)
        # print('S1 > ')
        return results

def predict_tflite(data, model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print('input_details ', input_details[0], ' data shape ', data.shape)
    interpreter.set_tensor(input_details[0]['index'], data)
    interpreter.invoke()
    landmarks = interpreter.get_tensor(output_details[0]['index'])
    print('landmarks = ', landmarks)
    return landmarks

def crop(img, box):
    return img[box.top(): box.bottom(), box.left(): box.right()]

def predict_single(img_path, model_path, 
                image_size=IMAGE_SIZE, 
                depth_multiplier=1.0,
                predict_fn=pfld_predict_landmarks,
                zero_mean=True,
                **kwargs):
    # get face bound
    img_size = image_size
    img = dlib.load_rgb_image(img_path)
    detector = dlib.get_frontal_face_detector()
    box = detector(img, 1)[0]
    oridata = cv2.imread(img_path)
    # if image_size ==80:
    #     oridata = cv2.cvtColor(oridata,cv2.COLOR_BGR2RGB)
    data = crop(oridata, box)
    data = resize(data, (img_size, img_size), anti_aliasing=True, mode='reflect') 
    # view_img(data, None)    
    normalized_data = normalize_data(data)
    if model_path.endswith('.tflite'):
        # print('using tflite model ', model_path)
        # is_unint8 = model_path.find('uint8') >= 0 
        # if is_unint8:
        #     print('int model')
        #     lmks = predict_tflite((np.reshape(data, (1, *data.shape)) * 255).astype(np.uint8), model_path)[0]
        # else:
        print('float model')
        
        lmks = predict_tflite(np.reshape(normalized_data, (1, *normalized_data.shape)).astype(np.float32), model_path)[0]
    else:
        lmks = predict(np.reshape(normalized_data, (1, *normalized_data.shape)), model_path, predict_fn,                   
                        image_size=image_size,
                        depth_multiplier=depth_multiplier,
                        **kwargs)[0]
    # print('landmark = ', lmks)
    if zero_mean:
        for i in range(0, 68):
            lmks[i*2] = (lmks[i*2]/2+0.5)* image_size# (lmks[i*2]/2+0.5)*image_size
            lmks[i*2+1] = (lmks[i*2+1]/2 + 0.5) * image_size# (lmks[i*2+1]/2 + 0.5)*image_size
        # print('landmarks after denorm', lmks)
    lmks = lmks.reshape((68, 2))

    view_img(data, lmks)

if __name__ == '__main__':
    # 2960256451_1.jpg
    # '/home/tamvm/Downloads/ibug_300W_large_face_landmark_dataset/helen/testset/30427236_1.jpg'
    use_tflite = False
    model = 'pfld-custom-80-025m-aux7'
    if model == 'pfld-64':
        predict_single('/home/tamvm/Downloads/test_face_tamvm_2.jpg', #'/home/tamvm/Downloads/ibug_300W_large_face_landmark_dataset/helen/trainset/2960256451_1.jpg', 
            '../../data/checkpoints-pfld-64-05m/pfld-311400' if not use_tflite else '../../data/pfld-64-quant.tflite',
            depth_multiplier=0.5,
            image_size=64)
    elif model == 'pfld-112':
        predict_single('/home/tamvm/Downloads/test_face_tamvm_2.jpg', #'/home/tamvm/Downloads/ibug_300W_large_face_landmark_dataset/helen/trainset/2960256451_1.jpg', 
            '../../data/checkpoints-pfld-112/pfld-1983600' if not use_tflite else '../../data/pfld-112-quant.tflite',
            # '../../data/pfld-64.tflite',
            image_size=112)
    elif model == 'pfld-80':
        predict_single('/home/tamvm/Downloads/test_face_tamvm_2.jpg', #'/home/tamvm/Downloads/ibug_300W_large_face_landmark_dataset/helen/trainset/2960256451_1.jpg', 
            '../../data/checkpoints-pfld-80-025m/pfld-333000',
            # '../../data/pfld-64.tflite',
            depth_multiplier=0.25,
            image_size=80)
    elif model == 'pfld-custom-80':
        predict_single('/home/tamvm/Downloads/test_face_tamvm_2.jpg', #'/home/tamvm/Downloads/ibug_300W_large_face_landmark_dataset/helen/trainset/2960256451_1.jpg', 
            '../../data/checkpoints-pfld-custom/pfld-183000',
            predict_fn=pfld_custom_predict_landmarks,
            # '../../data/pfld-64.tflite',
            depth_multiplier=1,
            zero_mean=True,
            image_size=80)
    elif model == 'pfld-custom-80-025m':
        predict_single('/home/tamvm/Downloads/test_face_tamvm_2.jpg', #'/home/tamvm/Downloads/ibug_300W_large_face_landmark_dataset/helen/trainset/2960256451_1.jpg', 
            '../../data/checkpoints-pfld-custom-80-025m/pfld-314100',
            predict_fn=pfld_custom_predict_landmarks,
            # '../../data/pfld-64.tflite',
            depth_multiplier=0.25,
            zero_mean=True,
            image_size=80)
    elif model == 'pfld-custom-80-025m-aux7':
        predict_single('/home/tamvm/Downloads/test_face_tamvm_2.jpg', #'/home/tamvm/Downloads/ibug_300W_large_face_landmark_dataset/helen/trainset/2960256451_1.jpg', 
            '../../data/checkpoints-pfld-custom-80-025m-aux7/pfld-292500',
            predict_fn=pfld_custom_predict_landmarks,
            # '../../data/pfld-64.tflite',
            depth_multiplier=0.25,
            zero_mean=True,
            image_size=80,
            aux_start_layer='layer_7')
    else:
        use_tflite = True
        predict_single('/home/tamvm/Downloads/test_face_tamvm_1.jpg', #'/home/tamvm/Downloads/ibug_300W_large_face_landmark_dataset/helen/trainset/2960256451_1.jpg', 
            '../../data/landmark_80pose.tflite',
            normalize_lmks=True,
            # '../../data/pfld-64.tflite',
            image_size=80)





import sys
sys.path.append('../../preprocess')

import numpy as np
import tensorflow as tf
from shapnet import predict_landmarks
from train import get_pcomps
from skimage.color import rgb2gray
import cv2
import dlib
from skimage.transform import resize
from prepare_data import IMAGE_SIZE, view_img
from skimage.transform import resize
import extractors
import matplotlib

matplotlib.use("TkAgg")
# IMAGE_SIZE = 224

def predict_tflite(data, model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print('input_details ', input_details[0], ' data shape ', data.shape)
    interpreter.set_tensor(input_details[0]['index'], data)
    interpreter.invoke()
    landmarks = interpreter.get_tensor(output_details[0]['index'])
    print('output_details', output_details)
    print('landmarks = ', landmarks)
    return landmarks

def predict(data, model_path, pca_path, image_size=IMAGE_SIZE, in_channels=1, feature_extractor=extractors.original_paper_feature_extractor):
    input_shape = [None, image_size, image_size] if in_channels == 1 else [None, image_size, image_size, in_channels]
    inputs = tf.placeholder(tf.float32, shape=input_shape, name='input_images')
    components = get_pcomps(pca_path)
    preds= predict_landmarks(inputs, components, feature_extractor=feature_extractor)    
    saver = tf.train.Saver()
    # g = tf.get_default_graph()
    # tf.contrib.quantize.create_eval_graph(input_graph=g)
    with tf.Session() as sess:         
        saver.restore(sess, model_path)
        # sess.run(tf.global_variables_initializer())
        results= sess.run(preds, feed_dict={inputs: data})
        print('landmarks = ', results)
        return results


def crop(img, box):
    return img[box.top(): box.bottom(), box.left(): box.right()]

# def normalize(data):
#     return (data - 0.5) * 2

def predict_single(img_path, model_path, pca_path, image_size=IMAGE_SIZE, 
                    feature_extractor=extractors.original_paper_feature_extractor,
                    in_channels=1):
    # get face bound
    img_size = image_size
    img = dlib.load_rgb_image(img_path)
    detector = dlib.get_frontal_face_detector()
    box = detector(img, 1)[0]
    oridata = cv2.imread(img_path)
    if in_channels == 1:
        oridata = rgb2gray(oridata)
    data = crop(oridata, box)
    data = resize(data, (img_size, img_size), anti_aliasing=True, mode='reflect') 
    # view_img(data, None)
    # data = normalize(data)
    if model_path.find('tflite') >= 0:
        print('predict using tflite ', model_path)
        lmks = predict_tflite(np.reshape(data, (1, *data.shape)).astype(np.float32), model_path)[0]
    else:
        lmks = predict(np.reshape(data, (1, *data.shape)), model_path, pca_path, 
                        feature_extractor=feature_extractor,
                        image_size=image_size, 
                        in_channels=in_channels)[0]
    # print('landmark = ', lmks)
    view_img(data, lmks)

if __name__ == '__main__':
    # 2960256451_1.jpg
    # '/home/tamvm/Downloads/ibug_300W_large_face_landmark_dataset/helen/testset/30427236_1.jpg'
    model = 'shapenet-224-1-depthwise'
    if model == 'shapnet-224-1':
        predict_single('/home/tamvm/Downloads/ibug_300W_large_face_landmark_dataset/helen/trainset/2960256451_1.jpg', 
            '../../data/checkpoints/shapenet-89000', 
            '../../data/unrot_train_pca.npz',
            image_size=224,
            in_channels=1)
    elif model == 'shapenet-224-1-depthwise':
        use_tflite = True
        predict_single('/home/tamvm/Downloads/test_face_tamvm_1.jpg',#'/home/tamvm/Downloads/ibug_300W_large_face_landmark_dataset/helen/trainset/2960256451_1.jpg', 
            '../../data/checkpoints-depthwise/shapenet-20500' if not use_tflite else '../../data/shapenet-depthwise.tflite', 
            '../../data/unrot_train_pca.npz',
            image_size=224,
            feature_extractor=extractors.depthwise_conv_feature_extractor,
            in_channels=1)
    # else:
    #     predict_single('/home/tamvm/Downloads/ibug_300W_large_face_landmark_dataset/helen/trainset/2960256451_1.jpg', 
    #         '../../data/checkpoints-64-v4/shapenet-94400', 
    #         '../../data/unrot_train_pca.npz',
    #         image_size=64,
    #         feature_extractor=extractors.mobilenet_extract_v4,
    #         in_channels=3)





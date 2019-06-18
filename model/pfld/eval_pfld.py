import sys
sys.path.append('../../preprocess')

import numpy as np
import tensorflow as tf
from pfld import predict_landmarks
from skimage.color import rgb2gray
import cv2
import dlib
from skimage.transform import resize
from prepare_data import IMAGE_SIZE, view_img
from skimage.transform import resize
import matplotlib
matplotlib.use("TkAgg")
# IMAGE_SIZE = 224

def predict(data, model_path, image_size=IMAGE_SIZE):
    input_shape = [None, image_size, image_size, 3]
    inputs = tf.placeholder(tf.float32, shape=input_shape, name='input_images')
    preds= predict_landmarks(inputs)    
    saver = tf.train.Saver()
    # g = tf.get_default_graph()
    # tf.contrib.quantize.create_eval_graph(input_graph=g)
    with tf.Session() as sess:         
        saver.restore(sess, model_path)
        # sess.run(tf.global_variables_initializer())
        results= sess.run(preds, feed_dict={inputs: data})
        return results


def crop(img, box):
    return img[box.top(): box.bottom(), box.left(): box.right()]

def normalize(data):
    return (data - 0.5) * 2

def predict_single(img_path, model_path, image_size=IMAGE_SIZE):
    # get face bound
    img_size = image_size
    img = dlib.load_rgb_image(img_path)
    detector = dlib.get_frontal_face_detector()
    box = detector(img, 1)[0]
    oridata = cv2.imread(img_path)

    data = crop(oridata, box)
    data = resize(data, (img_size, img_size), anti_aliasing=True, mode='reflect') 
    # view_img(data, None)
    data = normalize(data)
    lmks = predict(np.reshape(data, (1, *data.shape)), model_path,                    
                    image_size=image_size)[0]
    # print('landmark = ', lmks)
    lmks = lmks.reshape((68, 2))
    view_img(data, lmks)

if __name__ == '__main__':
    # 2960256451_1.jpg
    # '/home/tamvm/Downloads/ibug_300W_large_face_landmark_dataset/helen/testset/30427236_1.jpg'
    model = 'pfld-64'
    if model == 'pfld-64':
        predict_single('/home/tamvm/Downloads/ibug_300W_large_face_landmark_dataset/helen/trainset/2960256451_1.jpg', 
            '../../data/checkpoints-pfld-64/pfld-218400',
            image_size=64)
        pass





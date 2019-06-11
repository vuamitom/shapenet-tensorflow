import sys
sys.path.append('../preprocess')

import numpy as np
import tensorflow as tf
from shapnet import predict_landmarks
from train import get_pcomps
from skimage.color import rgb2gray
import cv2
import dlib
from matplotlib import pyplot as plt
from skimage.transform import resize
from prepare_data import IMAGE_SIZE, view_img
from skimage.transform import resize
# IMAGE_SIZE = 224

def predict(data, model_path, pca_path):
    image_size = IMAGE_SIZE
    inputs = tf.placeholder(tf.float32, shape=[None, image_size, image_size], name='input_images')
    components = get_pcomps(pca_path)
    preds = predict_landmarks(inputs, components)    
    saver = tf.train.Saver()
    with tf.Session() as sess:         
        saver.restore(sess, model_path)
        results = sess.run(preds, feed_dict={inputs: data})
        return results

def crop(img, box):
    return img[box.top(): box.bottom(), box.left(): box.right()]

def predict_single(img_path, model_path, pca_path):
    # get face bound
    img_size = IMAGE_SIZE
    img = dlib.load_rgb_image(img_path)
    detector = dlib.get_frontal_face_detector()
    box = detector(img, 1)[0]
    oridata = rgb2gray(cv2.imread(img_path))
    data = crop(oridata, box)
    data = resize(data, (img_size, img_size), anti_aliasing=True, mode='reflect') 
    lmks = predict(np.reshape(data, (1, *data.shape)), model_path, pca_path)[0]
    view_img(data, lmks)

if __name__ == '__main__':
    predict_single('/home/tamvm/Downloads/ibug_300W_large_face_landmark_dataset/helen/testset/30427236_1.jpg', 
        '../data/checkpoints/shapenet-89000', 
        '../data/unrot_train_pca.npz')




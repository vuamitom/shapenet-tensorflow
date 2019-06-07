from make_pca import load_landmarks
import numpy as np
import cv2 
from skimage.color import rgb2gray
from skimage.transform import AffineTransform, warp, resize
import os 
from matplotlib import pyplot as plt
import random
# import matplotlib.patches as patches
IMAGE_SIZE = 224
CROP_OFFSET = 0.05

def rotate(img, landmark, angle):
    h, w = img.shape[0], img.shape[1]
    center = (w/2, h/2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1)
    rotated_img = cv2.warpAffine(img, rot_mat, (w, h))
    rotated_landmarks = np.asarray([(int(rot_mat[0][0]*x+rot_mat[0][1]*y+rot_mat[0][2]),
                 int(rot_mat[1][0]*x+rot_mat[1][1]*y+rot_mat[1][2])) for (x, y) in landmark])
    return rotated_img, rotated_landmarks

def safe_rotate(img, landmark, angle):
    """
    fallback to a rotation that is not out of bound
    """
    rimg, rlmk = rotate(img, landmark, angle)
    h, w = rimg.shape[0], rimg.shape[1]
    for lm in rlmk:
        x, y = lm 
        if x < 0 or x > w or y < 0 or y > h:
            # print('rotation went out of frame when rotate', angle , 'fallback to rotate only ', fallback_angle, 'x vs w, y vs h', x, w, y, h)
            # return rotate(img, landmark, fallback_angle)
            return img, landmark
    return rimg, rlmk

def flip_indices():
    """
    refer to https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/
    """
    r = []
    for i in range(0, 8):
        r.append([i, 16 - i])
    # forhead
    for i in range(17, 22):
        r.append([i, 43 - i])
    # upper eyes
    for i in range(36, 40):
        r.append([i, 81 - i])

    # lower eyes
    r.append([41, 46])
    r.append([40, 47])

    # nose
    r.append([31, 35])
    r.append([32, 34])

    # lips
    r.append([48, 54])
    r.append([49, 53])
    r.append([50, 52])
    r.append([60, 64])
    r.append([61, 63])
    r.append([59, 55])
    r.append([67, 65])
    r.append([58, 56])
    return r


def flip(img, landmark):
    flipped_img = cv2.flip(img, 1)
    _, w = img.shape 
    flipped_landmark = np.asarray([(w-x, y) for (x, y) in landmark])
    for idx in flip_indices():
        flipped_landmark[idx] = flipped_landmark[[idx[1], idx[0]]]
    return flipped_img, flipped_landmark

def load_img(img_path):
    """read data from a single image. crop and rotate if necessary"""
    # read image
    img = cv2.imread(img_path)
    # crop
    return img

def crop(img, lmks, crop_offset=CROP_OFFSET):
    
    min_y, max_y = lmks[:,1].min(), lmks[:,1].max()
    min_x, max_x = lmks[:,0].min(), lmks[:,0].max() 
    # crop img data
    offset = int((max_x - min_x) * CROP_OFFSET)
    min_y, max_y = min_y - offset, max_y + offset
    min_x, max_x = min_x - offset, max_x + offset
    min_y = min_y if min_y > 0 else 0
    min_x = min_x if min_x > 0 else 0
    max_x = max_x if max_x < img.shape[1] else img.shape[1]
    max_y = max_y if max_y < img.shape[0] else img.shape[0]
    # print ('crop bound ', min_y, min_x, (max_x - min_x), (max_y - min_y))
    img = img[min_y:max_y, min_x:max_x]
    # crop lmks
    lmks = lmks - np.array([min_x, min_y])
    return img, lmks

def grayscale(img):
    return rgb2gray(img)# .reshape(img.shape[:-1], 1)

def view_img(img, lmks, ref_lmks = None):
    plt.imshow(img, cmap="gray")
    # top, left, w, h = bound
    # p = patches.Rectangle((top,left),w, h,linewidth=1,edgecolor='r',facecolor='none')
    plt.scatter(lmks[:, 0], lmks[:, 1], c="C0", s=15)
    # plt.add_patch(p)
    if ref_lmks is not None:
        plt.scatter(ref_lmks[:, 0], ref_lmks[:, 1], c="C1", s=15)
    plt.show()

def resize_lmks(img, lmks, img_size):
    target_shape = (img_size, img_size)
    # print('target_shape', target_shape, 'image shape ', img.shape[:-1], ' file name', name)
    scale = np.asarray(target_shape) / np.asarray(img.shape[:2])
    # print('scale = ', scale)
    trafo = AffineTransform(scale=scale)
    # img = warp(np.ascontiguousarray(img), trafo.inverse, output_shape=target_shape)
    lmks = trafo(np.ascontiguousarray(lmks[:, [1, 0]]))[:, [1, 0]]
    # lmks = warp(np.ascontiguousarray(lmks[:, [1, 0]]), trafo.inverse, output_shape=target_shape)[:, [1, 0]]
    return lmks

# def ensure_lmk_in_bound(lmks, w, h):
def read_data(lmk_xml, augment = True):    
    base_dir = os.path.dirname(lmk_xml)
    points, img_sizes, imgs = load_landmarks(lmk_xml)    
    img_size = IMAGE_SIZE
    no_augmented = 4
    data = np.ndarray((len(imgs) * no_augmented, img_size, img_size), dtype=np.float32)
    labels = np.ndarray((len(imgs)* no_augmented, *points[0].shape), dtype=np.int32)

    def crop_and_resize(img, lmks, img_size):
        img, lmks = crop(img, lmks)
        # view_img(img, lmks)
        lmks = resize_lmks(img, lmks, img_size)        
        img   = resize(img, (img_size, img_size), anti_aliasing=True, mode='reflect') 
        return img, lmks

    for i in range(0, len(imgs)):        
        # if not i == 1: continue
        img_path = os.path.join(base_dir, imgs[i])
        _, _, bound= img_sizes[i]

        original_im = grayscale(load_img(img_path))
        original_lmks = points[i]        
        # view_img(original_im, original_lmks)
        img, lmks = crop_and_resize(original_im, original_lmks, img_size)        
        fl_img, fl_lmks = flip(img, lmks)        
        # view_img(img, lmks)

        # print ('im size = ', im.shape, ' original ', original_im.shape)
        # make sure that face is at the center
        # original_im, original_lmks = crop(original_im, original_lmks, 0.4)
        rot_img, rot_lmk = safe_rotate(original_im, original_lmks, random.choice([5, 10, 15, 20]))        
        # view_img(rot_img, rot_lmk)
        rot_img, rot_lmk = crop_and_resize(rot_img, rot_lmk, img_size)

        rot_img_ccw, rot_lmk_ccw = safe_rotate(original_im, original_lmks, random.choice([-5, -10, -15, -20]))        
        # view_img(rot_img, rot_lmk)
        rot_img_ccw, rot_lmk_ccw = crop_and_resize(rot_img_ccw, rot_lmk_ccw, img_size)


        # view_img(rot_img, rot_lmk)
        for idx, gen_img in enumerate([(img, lmks), (fl_img, fl_lmks), (rot_img, rot_lmk), (rot_img_ccw, rot_lmk_ccw)]):
            data[i * no_augmented + idx] = gen_img[0]
            labels[i * no_augmented + idx] = gen_img[1]
            # w, h = gen_img[0].shape[1], gen_img[0].shape[0]
            # for lm in gen_img[1]:
            #     x, y = lm 
            #     if x < 0 or x > w or y < 0 or y > h:
            #         print('x vs w, y vs h', x, w, y, h, 'at pos ', idx)
            #         break
            # view_img(gen_img[0], gen_img[1])
        # return None
        if i % 300 == 0:
            print('processed ', (i+1), '/', len(imgs), ' images')
    return data, labels

def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,:,:]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

def preprocess(lmk_xml, output_dir):
    # save
    save_f = os.path.join(output_dir, os.path.basename(lmk_xml).replace('.xml', '.npz'))
    if os.path.exists(save_f):
        print ('preprocessed file exist: ', save_f)
        return

    data, labels = read_data(lmk_xml)
    # shuffle
    data, labels = randomize(data, labels)    
    # visualize to test after randomize
    # view_img(data[1], labels[1])
    np.savez_compressed(save_f, data=data, labels=labels)


if __name__ == '__main__':
    # preprocess train data
    preprocess('/home/tamvm/Downloads/ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train.xml', '../data')
    # preprocess test data
    preprocess('/home/tamvm/Downloads/ibug_300W_large_face_landmark_dataset/labels_ibug_300W_test.xml', '../data')
    # t = '/home/tamvm/Downloads/ibug_300W_large_face_landmark_dataset/helen/trainset/245871800_1.jpg'
    # im = load_img(t)
    # print (im.shape)
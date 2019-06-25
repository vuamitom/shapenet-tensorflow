from make_pca import load_landmarks
import numpy as np
import cv2 
from skimage.color import rgb2gray
from skimage.transform import AffineTransform, warp, resize
import os 
from matplotlib import pyplot as plt
import random
from pose_estimation import face_orientation, preview
from skimage import img_as_ubyte
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
    w = img.shape[1]
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

def crop(img, lmks, crop_offset=CROP_OFFSET, is_gray=True):
    
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
    if is_gray:
        img = img[min_y:max_y, min_x:max_x]
    else:
        img = img[min_y:max_y, min_x:max_x, :]
    # crop lmks
    lmks = lmks - np.array([min_x, min_y])
    return img, lmks

def grayscale(img):
    return rgb2gray(img)# .reshape(img.shape[:-1], 1)

def view_img(img, lmks, ref_lmks = None):
    is_gray = len(img.shape) == 2
    if is_gray:
        # print('abc----------->')
        plt.imshow(img, cmap="gray")
    else:
        temp =cv2.cvtColor(img.astype('float32'), cv2.COLOR_BGR2RGB)
        # print(img)
        plt.imshow(temp)
    # top, left, w, h = bound
    # p = patches.Rectangle((top,left),w, h,linewidth=1,edgecolor='r',facecolor='none')
    if lmks is not None:
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

def estimate_pose(frame, landmarks):
    
    # print(frame)
    imgpts, modelpts, rotate_degree, nose = face_orientation(frame, landmarks)
    # frame = img_as_ubyte(frame)
    # preview(frame, landmarks, rotate_degree, nose, imgpts, modelpts)
    return np.array(list(rotate_degree))
    # return rotate_degree

# def ensure_lmk_in_bound(lmks, w, h):
def read_data(lmk_xml, img_size, to_grayscale=True, rotate=True, include_pose=True):    
    base_dir = os.path.dirname(lmk_xml)
    points, img_sizes, imgs = load_landmarks(lmk_xml)    
    # img_size = IMAGE_SIZE
    if rotate:
        no_augmented = 3
    else:
        no_augmented = 2
    if to_grayscale:
        data = np.ndarray((len(imgs) * no_augmented, img_size, img_size), dtype=np.float32)
    else:
        data = np.ndarray((len(imgs) * no_augmented, img_size, img_size, 3), dtype=np.float32)
    labels = np.ndarray((len(imgs)* no_augmented, *points[0].shape), dtype=np.int32)
    pose_labels = None 
    if include_pose:
        pose_labels = np.ndarray((len(imgs)* no_augmented, 3), dtype=np.int32)

    def crop_and_resize(img, lmks, img_size):
        img, lmks = crop(img, lmks, is_gray=to_grayscale)
        # view_img(img, lmks)
        lmks = resize_lmks(img, lmks, img_size)        
        img   = resize(img, (img_size, img_size), anti_aliasing=True, mode='reflect') 
        return img, lmks

    for i in range(0, len(imgs)):        
        # if not i == 1: continue
        img_path = os.path.join(base_dir, imgs[i])
        _, _, bound= img_sizes[i]

        original_im = load_img(img_path)
        if to_grayscale:
            original_im = grayscale(original_im)

        original_lmks = points[i]        
        # view_img(original_im, original_lmks)
        img, lmks = crop_and_resize(original_im, original_lmks, img_size)   
        poses = None if not include_pose else estimate_pose(img, lmks)
        # view_img(img, lmks)     
        fl_img, fl_lmks = flip(img, lmks)        
        fl_poses = None if not include_pose else estimate_pose(fl_img, fl_lmks)
        # view_img(fl_img, fl_lmks)

        # print ('im size = ', im.shape, ' original ', original_im.shape)
        # make sure that face is at the center
        # original_im, original_lmks = crop(original_im, original_lmks, 0.4)
        gen_imgs = [(img, lmks, poses), (fl_img, fl_lmks, fl_poses)]
        if rotate:
            rot_img, rot_lmk = safe_rotate(original_im, original_lmks, random.choice([5, 10, 15, 20, 30, -5, -10, -15, -20, -30]))                    
            rot_img, rot_lmk = crop_and_resize(rot_img, rot_lmk, img_size)
            rot_poses = None if not include_pose else estimate_pose(rot_img, rot_lmk)
            gen_imgs.append((rot_img, rot_lmk, rot_poses))
        # view_img(rot_img, rot_lmk)
        # rot_img_ccw, rot_lmk_ccw = safe_rotate(original_im, original_lmks, random.choice([-5, -10, -15, -20]))        
        # view_img(rot_img, rot_lmk, is_gray=to_grayscale)
        # rot_img_ccw, rot_lmk_ccw = crop_and_resize(rot_img_ccw, rot_lmk_ccw, img_size)
        # view_img(rot_img, rot_lmk, is_gray=to_grayscale)
        for idx, gen_img in enumerate(gen_imgs):            
            data[i * no_augmented + idx] = gen_img[0]
            labels[i * no_augmented + idx] = gen_img[1]
            if include_pose:
                # print(gen_img[2])
                pose_labels[i* no_augmented + idx] = gen_img[2]

        # return None
        if i % 300 == 0:
            print('processed ', (i+1), '/', len(imgs), ' images')
    return data, labels, pose_labels

def randomize(dataset, labels, poses):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,:,:]
    shuffled_labels = labels[permutation]
    shuffled_poses = None
    if poses is not None:
        shuffled_poses = poses[permutation]
    return shuffled_dataset, shuffled_labels, shuffled_poses

def preprocess(lmk_xml, output_dir, image_size=IMAGE_SIZE, to_grayscale=True, include_pose=True):
    # save
    output_fn = os.path.basename(lmk_xml).replace('.xml', '_%s%s.npz' % (image_size, '_grey' if to_grayscale else ''))
    save_f = os.path.join(output_dir, output_fn)
    if os.path.exists(save_f):
        print ('preprocessed file exist: ', save_f)
        return

    data, labels, poses = read_data(lmk_xml, 
                                    image_size, 
                                    to_grayscale=to_grayscale, 
                                    include_pose=include_pose)
    # shuffle
    data, labels, poses = randomize(data, labels, poses)    
    # visualize to test after randomize
    # view_img(data[1], labels[1])
    np.savez_compressed(save_f, data=data, labels=labels, poses=poses)

def verify_data(path):
    img, lmks, poses = None, None, None
    with np.load(path) as ds:
        img = ds['data'][1]
        lmks = ds['labels'][1]
        poses = ds['poses'][1]

    frame = img_as_ubyte(img)    
    imgpts, modelpts, rotate_degree, nose = face_orientation(frame, lmks)
    print('diff = ', np.array(rotate_degree) - np.array(poses))
    preview(frame, lmks, rotate_degree, nose, imgpts, modelpts)


if __name__ == '__main__':
    # preprocess train data
    to_verify = False

    if to_verify:
        verify_data('../data/labels_ibug_300W_train_112.npz')
    else:
        image_size = 112
        to_grayscale = False
        gen_val_data = False
        gen_train_data = True
        include_pose = True

        if gen_train_data:
            preprocess('/home/tamvm/Downloads/ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train.xml', 
                        '../data', 
                        image_size=image_size,
                        include_pose=include_pose,
                        to_grayscale=to_grayscale)
        # preprocess test data,
        if gen_val_data:
            preprocess('/home/tamvm/Downloads/ibug_300W_large_face_landmark_dataset/labels_ibug_300W_test.xml', 
                        '../data', 
                        image_size=image_size,
                        include_pose=include_pose,
                        to_grayscale=to_grayscale)
        # t = '/home/tamvm/Downloads/ibug_300W_large_face_landmark_dataset/helen/trainset/245871800_1.jpg'
        # im = load_img(t)
        # print (im.shape)
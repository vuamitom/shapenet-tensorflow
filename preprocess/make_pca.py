import os
from copy import deepcopy
import numpy as np
from sklearn.decomposition import PCA
from skimage.transform import AffineTransform
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

####################################################
# image transformation method
# Credit: https://github.com/justusschock/shapedata/
####################################################

def get_angle(v0, v1, v2, degree=False):
    a1 = v0 - v1
    a2 = v0 - v2
    cosine_angle = np.dot(a1, a2) / (np.linalg.norm(a1) *
                                     np.linalg.norm(a2))

    angle = np.arccos(cosine_angle)

    if degree:
        angle = np.rad2deg(angle)

    return angle


def _normalize_rotation(lmks, img_size, index_left, index_right):

    left = lmks[index_left]
    right = lmks[index_right]

    diff = left - right

    middle = right + diff / 2

    length_middle_left = np.sqrt(((left - middle) ** 2).sum())

    left_optim = deepcopy(middle)
    left_optim[-1] += length_middle_left
    # print ('left', left, ' middle ', middle, ' left_optim ', left_optim)
    # get rotation angle
    rot_angle = get_angle(middle, left_optim, left, degree=False)

    affine_trafo = AffineTransform(rotation=rot_angle)

    shift_x, shift_y = np.array(img_size) / 2.

    # transform to shift image to origin
    tf_shift = AffineTransform(translation=[-shift_x, -shift_y])

    # transform to shift image back to original position
    tf_shift_inv = AffineTransform(translation=[shift_x, shift_y])

    complete_trafo = AffineTransform((tf_shift + (affine_trafo + tf_shift_inv)).params)

    r = complete_trafo(np.ascontiguousarray(lmks[:, [1, 0]]))[:, [1, 0]]
    # print ('new left ', r[index_left], ' new right ', r[index_right])
    return r
    # return complete_trafo(np.ascontiguousarray(lmks)

############################################################    

def make_pca(landmarks, scale = True, center = True):
    """
    perform PCA on samples' landmarks
    Parameters
    ----------
    scale : bool
        whether or not to scale the principal components with the
        corresponding eigen value
    center : bool
        whether or not to substract mean before pca
    Returns
    -------
    np.array
        eigen_shapes
    """
    if center:
        mean = np.mean(landmarks.reshape(-1, landmarks.shape[-1]), axis=0)
        landmarks = landmarks - mean
    landmarks_transposed = landmarks.transpose((0, 2, 1))

    reshaped = landmarks_transposed.reshape(landmarks.shape[0], -1)

    pca = PCA()
    pca.fit(reshaped)

    if scale:
        components = pca.components_ * pca.singular_values_.reshape(-1, 1)
    else:
        components = pca.components_

    return np.array([pca.mean_] + list(components)).reshape(
        components.shape[0] + 1,
        *landmarks_transposed.shape[1:]).transpose(0, 2, 1)

def view_landmarks(lmks, img = None):
    """show landmark"""
    # print(img)
    marker_size = 15    
    if img is not None and False:
        plt.imshow(mpimg.imread(img))
    else:
        # convert to image origin
        plt.gca().invert_yaxis()
    plt.scatter(lmks[:, 0], lmks[:, 1], c="C0", s=marker_size)
    plt.show()

def normalize_rotation(lmks, img_size, rotation_idxs=(37, 46)):    
    for idx in range(len(lmks)):
        # plot 
        w, h, _ = img_size[idx]
        lmks[idx] = _normalize_rotation(lmks[idx], (w, h), rotation_idxs[0], rotation_idxs[1])
    return lmks

def load_landmarks(lmk_xml):
    assert lmk_xml.endswith('.xml')
    print('loading landmarks... ', lmk_xml)    
    
    import xml.etree.ElementTree as ET
    from PIL import Image 
    tree = ET.parse(lmk_xml)
    root = tree.getroot()
    base = os.path.dirname(lmk_xml)

    points = []
    img_shapes = []
    img_names = []
    for img in root.iter('image'):
        box = img.find('box')
        path = os.path.join(base, img.get('file'))
        xy = []
        for p in box.iter('part'):
            # xs.append(int(p.get('x')))
            # ys.append(int(p.get('y')))
            xy.append((int(p.get('x')), int(p.get('y'))))
        # xs = np.array(xs, dtype=np.int32).reshape((-1, 1))
        # ys = np.array(ys, dtype=np.int32).reshape((-1, 1))
        # points.append(np.hstack([xs, ys]))
        points.append(np.array(xy, dtype=np.int32))
        img_names.append(img.get('file'))
        bound = (int(box.get('top')), int(box.get('left')), int(box.get('width')), int(box.get('height')))
        with Image.open(path) as imgRef:
            w, h = imgRef.size
            img_shapes.append((w, h, bound))

    return np.array(points), img_shapes, img_names

def prepare_dlib_dset(lmk_xml, output_dir, normalize_rot=False):
    """
    set normalize_rot to False. When it's True, I faced some issue with align eye landmarks

    Prepare datasets similar to one available in dlib.
    Download according to this instruction 
    https://medium.com/datadriveninvestor/training-alternative-dlib-shape-predictor-models-using-python-d1d8f8bd9f5c
    """
    # base_dir = os.path.dirname(lmk_xml)
    pca_file = os.path.join(output_dir, 'train_pca.npz' if normalize_rot else 'unrot_train_pca.npz')
    if not os.path.exists(pca_file):
        points, img_sizes, imgs = load_landmarks(lmk_xml)
        # view_landmarks(points[20], os.path.join(base_dir, imgs[20]), img_sizes[20])
        if normalize_rot:
            points = normalize_rotation(points, img_sizes)
        # view_landmarks(points[20])
        pca = make_pca(points, img_sizes)        
        print('pca size = ', pca.shape)
        # print(pca[0])
        np.savez(pca_file, shapes=pca)
    else:
        print ('pca file already exists at ', pca_file)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        help="Path to dataset dir",
                        type=str)
    parser.add_argument("--output",
                        help="Path to output dir",
                        type=str)
    args = parser.parse_args()
    data_dir = args.dataset
    output_dir = args.output
    prepare_dlib_dset(data_dir if data_dir is not None 
        else '/home/tamvm/Downloads/ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train.xml',
        output_dir if output_dir is not None
        else '../data')
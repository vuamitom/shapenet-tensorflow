# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 11:58:14 2018
https://github.com/jerryhouuu/Face-Yaw-Roll-Pitch-from-Pose-Estimation-using-OpenCV
@author: jerry
"""
from make_pca import load_landmarks
import cv2
import math
import numpy as np
import os



NOSE = 31 - 1
CHIN = 9 - 1
LEFT_EYE = 37 - 1
RIGHT_EYE = 46 -1 
LEFT_MOUTH = 49 -1
RIGHT_MOUTH = 55 -1

def face_orientation(frame, landmarks):
    size = frame.shape #(height, width, color_channel)

    image_points = np.array([
                            (landmarks[NOSE][0], landmarks[NOSE][1]),     # Nose tip
                            (landmarks[CHIN][0], landmarks[CHIN][1]),       # Chin
                            (landmarks[LEFT_EYE][0], landmarks[LEFT_EYE][1]),     # Left eye left corner
                            (landmarks[RIGHT_EYE][0], landmarks[RIGHT_EYE][1]),     # Right eye right corne
                            (landmarks[LEFT_MOUTH][0], landmarks[LEFT_MOUTH][1]),     # Left Mouth corner
                            (landmarks[RIGHT_MOUTH][0], landmarks[RIGHT_MOUTH][1])      # Right mouth corner
                        ], dtype=np.float32)
                        
    model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            ( 225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner                         
                        ])

    # Camera internals
 
    center = (size[1]/2, size[0]/2)
    focal_length = center[0] / np.tan(60/2 * np.pi / 180)
    camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = np.float32
                         )

    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, 
                                                                image_points, 
                                                                camera_matrix, 
                                                                dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    
    axis = np.float32([[500,0,0], 
                          [0,500,0], 
                          [0,0,500]])
                          
    imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    modelpts, jac2 = cv2.projectPoints(model_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]

    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6] 

    
    pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]


    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))

    return imgpts, modelpts, (str(int(roll)), str(int(pitch)), str(int(yaw))), (landmarks[NOSE][0], landmarks[NOSE][1])

# f = open('/home/jerry/Documents/test/test/landmark.txt','r')
# for line in iter(f):
#     img_info = line.split(' ')
#     img_path = img_info[0]
#     frame = cv2.imread(img_path)
#     landmarks =  map(int, img_info[1:])
lmk_xml = '/home/tamvm/Downloads/ibug_300W_large_face_landmark_dataset/labels_ibug_300W_test.xml'
points, img_sizes, imgs = load_landmarks(lmk_xml)
base_dir = os.path.dirname(lmk_xml)
for i in range(0, 10):        
    # if not i == 1: continue
    img_path = os.path.join(base_dir, imgs[i])
    # original_im = load_img(img_path)
    landmarks = points[i]
    frame = cv2.imread(img_path)

    
    imgpts, modelpts, rotate_degree, nose = face_orientation(frame, landmarks)
    print(img_path, ' ', imgpts)

    cv2.line(frame, nose, tuple(imgpts[1].ravel()), (0,255,0), 3) #GREEN
    cv2.line(frame, nose, tuple(imgpts[0].ravel()), (255,0,), 3) #BLUE
    cv2.line(frame, nose, tuple(imgpts[2].ravel()), (0,0,255), 3) #RED
    
    remapping = [2,3,0,4,5,1]
    selected_landmarks = [LEFT_EYE, RIGHT_EYE, NOSE, LEFT_MOUTH, RIGHT_MOUTH, CHIN]
    for index in range(0, len(selected_landmarks)):
        random_color = tuple([int(x) for x in (np.random.random_integers(0,255,size=3))])
        # print('random_color', type(random_color[0]), type((20,60,80)[0]))
        cv2.circle(frame, (landmarks[selected_landmarks[index]][0], landmarks[selected_landmarks[index]][1]), 5, random_color, -1)  
        cv2.circle(frame,  tuple(modelpts[remapping[index]].ravel().astype(int)), 2, random_color, -1)  
        
            
#    cv2.putText(frame, rotate_degree[0]+' '+rotate_degree[1]+' '+rotate_degree[2], (10, 30),
#                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
#                thickness=2, lineType=2)
                
    for j in range(0, len(rotate_degree)):
        cv2.putText(frame, ('{:05.2f}').format(float(rotate_degree[j])), (10, 30 + (50 * j)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=2)

    cv2.imwrite('/home/tamvm/Projects/shapenet-tensorflow/preprocess/'+ os.path.basename(img_path), frame)

# f.close()

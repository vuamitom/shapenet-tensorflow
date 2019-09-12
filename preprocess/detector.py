from PIL import Image
import numpy as np
import tensorflow as tf
import time
from skimage.transform import AffineTransform, warp
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import cv2

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


class Detector:
    def __init__(self):
        model_path = '/home/tamvm/AndroidStudioProjects/MLKitDemo/app/src/main/assets/detect_face_48_quantized_lite.tflite'
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()    
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.crop_size = 48

    def _detect(self, image_np_expanded, ori_size, oriimage=None):
        self.interpreter.set_tensor(self.input_details[0]['index'], image_np_expanded)
        t0 = time.clock()
        self.interpreter.invoke()
        # print('num_detection = ', output_dict['num_boxes'])
        # box = output_dict['detection_boxes'][0]
        box = self.interpreter.get_tensor(self.output_details[0]['index'])[0][0]
        score = self.interpreter.get_tensor(self.output_details[2]['index'])[0][0]
        if score < 0.51:
            # print('low score, return none')
            return None 
        # score = output_dict['detection_scores'][0]
        print('returning box ', box, ' score = ', score)
        # return None
        # return box 
        # left, top, right, bottom 
        crop_size = image_np_expanded.shape[1]
        left, top, right, bottom = box[1]*crop_size, box[0] * crop_size, box[3]*crop_size, box[2] * crop_size
        points = np.array([[left, top], [right, bottom]])
        ori_w, ori_h = ori_size
        points = resize(image_np_expanded.shape[1:3], points, (ori_h, ori_w))

        if oriimage is not None:
            fig,ax = plt.subplots(1)
            ax.imshow(oriimage)
            print('detect points = ', points)
            # p = patches.Rectangle((left, top), right - left, bottom - top, linewidth=1, edgecolor='r',facecolor='none')
            p = patches.Rectangle((points[0][0], points[0][1]), points[1][0] - points[0][0], points[1][1] - points[0][1], linewidth=1, edgecolor='r',facecolor='none')
            ax.add_patch(p)
            plt.show()

        return int(points[0][0]), int(points[0][1]), int(points[1][0]), int(points[1][1])

    def detect_cv(self, oriim):
        # oriim = cv2.imread(image_path)
        print('cv_shape = ',oriim.shape)
        crop_size = self.crop_size
        im = cv2.resize(oriim, (crop_size, crop_size))
        return self._detect(np.expand_dims(im, axis=0), (oriim.shape[1], oriim.shape[0]))

    def detect(self, image_path):
        crop_size = self.crop_size
        oriimage = Image.open(image_path)
        image = oriimage.resize((crop_size, crop_size), Image.ANTIALIAS)
        image_np = load_image_into_numpy_array(image)
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # print('image_shape', image_np_expanded.shape)
        return self._detect(image_np_expanded, oriimage.size)
        # output_dict = tflite_run_inference_for_single_image(image_np_expanded)

        


# def tflite_run_inference_for_single_image(image):
#     # Load TFLite model and allocate tensors.
#     # model_path = '/home/tamvm/AndroidStudioProjects/MLKitDemo/app/src/main/assets/detect_face_224_quantized.tflite'
#     model_path = '/home/tamvm/AndroidStudioProjects/MLKitDemo/app/src/main/assets/detect_face_48_quantized_lite.tflite'
#     interpreter = tf.lite.Interpreter(model_path=model_path)
#     interpreter.allocate_tensors()

#     # Get input and output tensors.
#     input_details = interpreter.get_input_details()
#     output_details = interpreter.get_output_details()
#     # print('input details = ', input_details)
#     # print('output details = ', input_details)
#     # Test model on random input data.
#     input_shape = input_details[0]['shape']
#     # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
#     input_data = image
#     # print('required shape = ', input_shape, ' actual = ', input_data.shape)
#     interpreter.set_tensor(input_details[0]['index'], input_data)
#     t0 = time.clock()
#     interpreter.invoke()
#     t1 = time.clock() - t0
#     # print('inference time = ', t1)
#     # TFLite_Detection_PostProcess custom op node has four outputs:
#     # detection_boxes: a float32 tensor of shape [1, num_boxes, 4] with box
#     # locations
#     # detection_classes: a float32 tensor of shape [1, num_boxes]
#     # with class indices
#     # detection_scores: a float32 tensor of shape [1, num_boxes]
#     # with class scores
#     # num_boxes: a float32 tensor of size 1 containing the number of detected boxes
#     output_dict = {
#         'detection_boxes': interpreter.get_tensor(output_details[0]['index'])[0],
#         'detection_classes': interpreter.get_tensor(output_details[1]['index'])[0],
#         'detection_scores': interpreter.get_tensor(output_details[2]['index'])[0],
#         'num_boxes': interpreter.get_tensor(output_details[3]['index'])[0]
#     }
#     # print(output_dict)
#     # print('no of detections = ', output_dict['num_boxes'])
#     output_dict['detection_classes'] = [int(x) for x in output_dict['detection_classes']]
#     return output_dict


def resize(img_shape, points, target_shape):
    # target_shape = (img_size, img_size)
    # print('target_shape', target_shape, 'image shape ', img.shape[:-1], ' file name', name)
    scale = np.asarray(target_shape) / np.asarray(img_shape)
    # print('scale = ', scale)
    trafo = AffineTransform(scale=scale)
    # img = warp(np.ascontiguousarray(img), trafo.inverse, output_shape=target_shape)
    points = trafo(np.ascontiguousarray(points[:, [1, 0]]))[:, [1, 0]]
    # lmks = warp(np.ascontiguousarray(lmks[:, [1, 0]]), trafo.inverse, output_shape=target_shape)[:, [1, 0]]
    return points


# def detect(image_path):
#     crop_size = 48
#     oriimage = Image.open(image_path)
#     image = oriimage.resize((crop_size, crop_size), Image.ANTIALIAS)
#     image_np = load_image_into_numpy_array(image)
#     image_np_expanded = np.expand_dims(image_np, axis=0)
#     output_dict = tflite_run_inference_for_single_image(image_np_expanded)
#     # print('num_detection = ', output_dict['num_boxes'])
#     box = output_dict['detection_boxes'][0]
#     score = output_dict['detection_scores'][0]
#     print('returning box ', box, ' with score ', score)
#     # return None
#     # return box 
#     # left, top, right, bottom 

#     left, top, right, bottom = box[1]*crop_size, box[0] * crop_size, box[3]*crop_size, box[2] * crop_size
#     points = np.array([[left, top], [right, bottom]])
#     ori_w, ori_h = oriimage.size
#     points = resize(image_np, points, (ori_h, ori_w))

#     if False:
#         fig,ax = plt.subplots(1)
#         ax.imshow(oriimage)
#         print('detect points = ', points)
#         # p = patches.Rectangle((left, top), right - left, bottom - top, linewidth=1, edgecolor='r',facecolor='none')
#         p = patches.Rectangle((points[0][0], points[0][1]), points[1][0] - points[0][0], points[1][1] - points[0][1], linewidth=1, edgecolor='r',facecolor='none')
#         ax.add_patch(p)
#         plt.show()

#     return int(points[0][0]), int(points[0][1]), int(points[1][0]), int(points[1][1])
#     # revert to size

def get_face_detector():
    return Detector()


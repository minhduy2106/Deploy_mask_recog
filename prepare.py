import torch
import cv2
import numpy as np
import sys
from model_detectface_and_recogmask.yolov7_face.models.experimental import attempt_load
from model_detectface_and_recogmask.yolov7_face.utils.datasets import letterbox
from model_detectface_and_recogmask.yolov7_face.utils.general import check_img_size, scale_coords, non_max_suppression
import time
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

sys.path.insert(0, './model_detectface_and_recogmask/yolov7_face')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_detect_face = attempt_load("model_detectface_and_recogmask/yolov7_face/best.pt", map_location=device)
INIT_LR = 1e-4
model_recog_mask = tf.keras.models.load_model('model_detectface_and_recogmask/recog_mask.h5', compile=False)
model_recog_mask.compile(loss="binary_crossentropy", optimizer=Adam(INIT_LR), metrics=["accuracy"])

########################################################
size_convert = 640
conf_thres = 0.4
iou_thres = 0.5
########################################################
#resize image
def resize_image(img0, img_size, orgimg):
    h0, w0 = orgimg.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s=model_detect_face.stride.max())  # check img_size
    img = letterbox(img0, new_shape=imgsz)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    return img
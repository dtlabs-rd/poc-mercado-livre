import numpy as np
import cv2

class ImagePreprocessor:

    def __init__(self):
        self.img = None
        self.img_resized = None
        self.resize_time_list = []

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
        if self.img is None:
            self.img = np.zeros(new_shape, dtype = np.float32)
        if self.img_resized is None:
            self.img_resized = np.zeros(new_shape, dtype = np.float32)
        # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232

        img_shape = img.shape
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        # only scale down, do not scale up (for better test mAP)
        if not scaleup:
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
            new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / \
                shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)

    def normalize(self, img):
        return np.float32(img / 255.0)

    def expand_dims(self, img):
        return np.expand_dims(img, axis=0)

    def bgr2rgb(self, img):
        return img[:, :, ::-1].transpose(2, 0, 1)

    def run(self, image, new_shape=(640, 640)):
        self.img = self.letterbox(image, new_shape=new_shape)[0]
        self.img = self.normalize(self.img)
        self.img = self.bgr2rgb(self.img)
        self.img = self.expand_dims(self.img)

        return self.img
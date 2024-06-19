import numpy as np
import cv2
from ocr import OCR
from object_detection import YoloV5Detector

Image = np.ndarray

class ALPR():
    def __init__(
        self, 
        det_weights,
        ocr_weights,
        det_conf_thres=0.2,
        det_iou_thres=0.2,
        ocr_conf_thres=0.2
        
    ):
        # Load object detector
        self.det = YoloV5Detector(
            det_weights,
            det_conf_thres,
            det_iou_thres
        )
        # Load ocr model
        self.ocr = OCR(
            ocr_weights,
            ocr_conf_thres
        )

    def prepare_image(self, image: Image | str) -> Image:
        """
        Preprocess an image.
        """

        # If the image is a path, read it
        if isinstance(image, str):
            image = cv2.imread(image)

        return image

    def _det(self, image):
        predictions = self.det.run(image)
        return predictions

    def _ocr(self, image):
        res, confs = self.ocr.infer_plate(image)
        return "".join(res)

    def _crop(self, image, dets):
        output_image_list = list()
        dets = np.maximum(dets, 0.0)
        for det in dets:
            x1, y1, x2, y2 = map(int, det[:4])
            output_image_list.append(image[y1:y2, x1:x2])
        return output_image_list
    
    def __call__(self, image: Image | str):
        """
        Call an ALPR pipeline on a single image
        """

        image = self.prepare_image(image)
        dets = self._det(image)
        crops = self._crop(image, dets)
        return [{"text":self._ocr(crops[i]), "detection": dets[i]} for i in range(len(crops))]
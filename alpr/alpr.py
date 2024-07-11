import numpy as np
import cv2
import string
from .ocr import OCR
from .object_detection import YoloV5Detector
from .sort import Tracker

Image = np.ndarray

class ALPR():
    def __init__(
        self, 
        det_weights,
        ocr_weights,
        det_conf_thres=0.2,
        det_iou_thres=0.2,
        ocr_conf_thres=0.2,
        use_tracker=False
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
        # Tracker
        self.use_tracker = use_tracker
        if use_tracker:
            self.tracker = Tracker(
                max_age=10,
                min_hits=3,
                iou_threshold=0.2
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
    
    def _post_process_ocr(self, plate: str) -> str:
        # Standards:
        # A A A X X X X
        # A A A X A X X
        #         ^- ambiguous

        letters = [A for A in string.ascii_uppercase]
        numbers = [str(X) for X in range(10)]
        # disambiguation rules
        replace_A = {"0": "O", "1": "I", "5": "S", "8": "B", "Z": "2"}
        replace_X = {"O": "0", "J": "1", "I": "1", "S": "5", "B": "8", "2": "Z"}

        result = ""
        for idx, char in enumerate(plate):
            if idx in range(0, 3) and char not in letters:
                char = replace_A.get(char, char)

            if idx in [3, 5, 6] and char not in numbers:
                char = replace_X.get(char, char)
            # PS: idx == 4 is ambiguous
            result += char

        return result

    def _ocr(self, image):
        res, confs = self.ocr.infer_plate(image)
        plate = "".join(res)
        return self._post_process_ocr(plate)

    def _crop(self, image, dets):
        output_image_list = list()
        for det in dets:
            det = np.maximum(det, 0.0)
            x1, y1, x2, y2 = map(int, det[:4])
            output_image_list.append(image[y1:y2, x1:x2])
        return output_image_list
    
    def __call__(self, image: Image | str):
        """
        Call an ALPR pipeline on a single image
        """

        image = self.prepare_image(image)
        dets = np.array(self._det(image)).reshape(-1, 6)
        
        if self.use_tracker:
            
            tracklets = self.tracker.update(dets)
            dets = [det.get_state()[0] for det in tracklets]
            crops = self._crop(image, dets)
            return [{
                "text":self._ocr(crops[i]), 
                "detection": dets[i],
                "id": tracklets[i].id
            } for i in range(len(dets))]
            
        else:
            
            crops = self._crop(image, dets)
            return [{
                "text":self._ocr(crops[i]), 
                "detection": dets[i],
            } for i in range(len(dets))]
        
        
if __name__ == "__main__":
    
    # Video capture
    cap = cv2.VideoCapture('../data/video_0_94.mp4')
        
    # ALPR pipeline
    alpr_pipeline = ALPR(
        det_weights='../models/alpr_detector.onnx',
        ocr_weights='../models/alpr_ocr.onnx'
    )
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        'output_1.mp4',  
        cv2.VideoWriter_fourcc(*'MJPG'), 
        fps, 
        (width, height)
    ) 

    while True:
        _, frame = cap.read()
        
        
        frame = alpr_pipeline.prepare_image(frame)
        results = alpr_pipeline(frame)
        
        for result in results:
            x1, y1, x2, y2 = map(int, result['detection'][:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0,255,0), thickness=2)
            cv2.putText(frame, result['text'], (x1, y1), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,255,0), thickness=2)

        cv2.imshow('frame', frame)
        writer.write(frame) 
            
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

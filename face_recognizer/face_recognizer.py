from typing import List
import cv2
import numpy as np
import math
from insightface.app import FaceAnalysis

Image = np.ndarray
np.int = np.int32
np.bool = bool

class Face:
    """
    A class representing a detected face in a single frame.
    """

    def __init__(self, face, eye_distance_thresh):
        self.face = face
        self.eye_distance_thresh = eye_distance_thresh

    def det_score(self):
        """
        Return the detection confidence of the face
        """

        return self.face.det_score

    def embedding(self):
        """
        Return a vector embedding of the face
        """
        return self.face.normed_embedding

    def area(self) -> float:
        """
        Find the area of the bounding box of the face in the original image.
        """

        bbox = self.face.bbox
        return abs(bbox[0] - bbox[2]) * abs(bbox[1] - bbox[3])
    
    def valid_angle(self) -> bool:
        """
        Check if the face is a valid detection using eyes distance.
        """
        l_eye, r_eye = self.face['kps'][:2]
        return math.dist(l_eye, r_eye) > self.eye_distance_thresh
    
class FaceRecognizer:
    """
    A class that consumes images and produces lists of Face objects using an insightface pipeline
    """

    def __init__(
        self,
        use_colors: bool = False,
        det_thresh: float = 0.4,
        nms_thresh: float = 0.4,
        eye_distance_thresh: float = 60,
    ):
        self.pipeline = FaceAnalysis(allowed_modules=['detection', 'recognition'])
        self.pipeline.prepare(ctx_id=0)
        self.use_colors = use_colors

        self.pipeline.models["detection"].det_thresh = det_thresh
        self.pipeline.models["detection"].nms_thresh = nms_thresh
        self.eye_distance_thresh = eye_distance_thresh

    def prepare_image(self, image: Image | str) -> Image:
        """
        Preprocess an image.
        """

        # If the image is a path, read it
        if isinstance(image, str):
            image = cv2.imread(image)

        # Unless colors are enabled, convert the image to grayscale
        if not self.use_colors:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = np.stack([image] * 3, axis=-1)

        return image

    def __call__(self, image: Image | str) -> List[Face]:
        """
        Call an insightface pipeline on a single image
        """

        image = self.prepare_image(image)
        faces = self.pipeline.get(image)
        return [Face(face, self.eye_distance_thresh) for face in faces]


if __name__ == "__main__":
    
    # Video capture
    cap = cv2.VideoCapture(1)
        
    # Face recognition pipeline
    face_recognition_pipeline = FaceRecognizer(use_colors=True)

    while True:
        _, frame = cap.read()
        
        frame = face_recognition_pipeline.prepare_image(frame)
        results = face_recognition_pipeline(frame)
        
        for result in results:
            color = (0,255,0) if result.valid_angle() else (0,0,255)
            # Draw bbox
            x1, y1, x2, y2 = map(int, result.face.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color)
            # Draw kpts
            for kpt in result.face.kps[:2]:
                x, y = map(int, kpt)
                cv2.circle(frame, (x, y), 2, color, thickness=2)

        cv2.imshow('frame', frame)
            
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break



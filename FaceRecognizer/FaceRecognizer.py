from typing import Iterator, List
import cv2
import numpy as np
import pickle
from insightface.app import FaceAnalysis

Image = np.ndarray
np.int = np.int32
np.bool = bool

def get_object(name):
    with open(name, "rb") as f:
        obj = pickle.load(f)
    return obj

class Face:
    """
    A class representing a detected face in a single frame.
    """
    
    # Static reference shape for face alignment
    reference_3d_kpts = get_object("../assets/meanshape_68.pkl")

    def __init__(self, face, angle_thresh: float = 45):
        self.face = face
        self.angle_thresh = angle_thresh

    def det_score(self):
        """
        Return the detection confidence of the face
        """

        return self.face.det_score

    def vector(self):
        """
        Return a vector embedding of the face
        """
        return self.face.normed_embedding

    def rot_matrix(self) -> np.ndarray:
        """
        Compute the optimal rotation matrix that aligns the given face
        with the reference 3D face shape.

        For this, we use the Kabsch algorithm, which uses the SVD of
        the covariance matrix of the two sets of points.
        """

        keypoints = self.face["landmark_3d_68"]
        reference = Face.reference_3d_kpts.copy()
        reference[:, 2] *= -1

        keypoints -= keypoints.mean(0)
        reference -= reference.mean(0)
        covariance = keypoints.T @ reference

        U, S, Vt = np.linalg.svd(covariance)
        R = U @ Vt
        return R

    def direction(self) -> np.ndarray:
        """
        Get a vector pointing to the direction that the face is looking at
        """

        # Return the third column of the rotation matrix
        R = self.rot_matrix()
        R = np.round(R, 3)
        return R[:, 2]

    def angle(self) -> float:
        """
        Return the angle between the face and the camera, in degrees.

        The angle of a face looking straight ahead at the camera is 0.
        """

        cos = self.direction() @ np.array([0, 0, 1])
        angle = np.arccos(cos) * 180 / np.pi
        return angle

    def area(self) -> float:
        """
        Find the area of the bounding box of the face in the original image.
        """

        bbox = self.face.bbox
        return abs(bbox[0] - bbox[2]) * abs(bbox[1] - bbox[3])

    def valid_angle(self) -> bool:
        """
        Check if the face is a valid detection using its angle.
        """

        return self.angle() < self.angle_thresh
    
class FaceRecognizer:
    """
    A class that consumes images and produces lists of Face objects using an insightface pipeline
    """

    def __init__(
        self,
        use_colors: bool = False,
        det_thresh: float = 0.6,
        nms_thresh: float = 0.4,
        angle_thresh: float = 45,
    ):
        self.pipeline = FaceAnalysis()
        self.pipeline.prepare(ctx_id=0)
        self.use_colors = use_colors

        # self.model.models["detection"].det_thresh = det_thresh
        # self.model.models["detection"].nms_thresh = nms_thresh
        # self.angle_thresh = angle_thresh

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
        return [Face(face, image) for face in faces]




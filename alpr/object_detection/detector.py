# import cupy
import onnxruntime
import numpy as np
from .nms import NonMaxSupression
from .rescale import Rescale
from .image_preprocessor import ImagePreprocessor
# from detector.models.TRTModel import BaseTensorRTModel


class NNBackbone:
    def __init__(self, weights):
        self.weight_type = weights.split('.')[-1].lower()

        if self.weight_type == 'onnx':
            self.nn = onnxruntime.InferenceSession(
                weights, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.input_shape = self.nn.get_inputs()[0].shape
        # elif self.weight_type == 'trt':
        #     self.nn = BaseTensorRTModel(model_path=weights, use_engine_cache=True)
        #     self.input_shape = self.nn.inputs[0].shape

    def __call__(self, image):
        if self.weight_type == 'onnx':
            return self.nn.run(None, {"images": image})[0]
        elif self.weight_type == 'trt':
            return cupy.asnumpy(self.nn(cupy.array(image))['output'])


class YoloV5Detector:
    """
    This class implements the necessary steps to perform the object detection.
    """

    def __init__(
        self,
        weights,
        conf_thres,
        iou_thres
    ):
        self.backbone = NNBackbone(weights)

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        self.input_shape = self.backbone.input_shape

        self.nms = NonMaxSupression()
        self.transform = Rescale()
        self.img_processor = ImagePreprocessor()
        self.image_0 = None
        self.image_1 = None
        self.image_height, self.image_width = None, None
        self.inference_size = (self.input_shape[2], self.input_shape[3])
        self.batch_size = self.input_shape[0]
        self.__warmup()

    def preprocess(self, image):
        img = self.img_processor.run(image, self.inference_size)
        return img

    def predict(self, image):
        self.pred_test = self.backbone(image)
        # only the first layer of pred is useful
        self.pred_test = self.nms.run(self.pred_test, self.conf_thres, self.iou_thres)
        return self.pred_test

    def rescale(self, image, pred):
        pred = self.transform.run(pred, image.shape, new_shape=self.inference_size)
        return pred

    def run(self, image):
        image_processed = self.preprocess(image)
        pred = self.predict(image_processed)
        if pred[0] is not None:
            pred[0] = self.rescale(image, pred[0])
        else:
            return []

        # returns the bbox position tl_x, tl_y, br_x, br_y, class, conf
        return pred[0]

    def __warmup(self):
        _ = self.run(np.random.rand(*self.inference_size, 3))

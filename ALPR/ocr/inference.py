import onnxruntime
import numpy as np
import glob
import cv2
# import cupy
from scipy.special import softmax

# from detector.models.TRTModel import BaseTensorRTModel


class NNBackbone:
    def __init__(self, weights):
        self.weight_type = weights.split('.')[-1].lower()

        if self.weight_type == 'onnx':
            self.nn = onnxruntime.InferenceSession(
                weights, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )

        # elif self.weight_type == 'trt':
        #     self.nn = BaseTensorRTModel(model_path=weights, use_engine_cache=True)

    def __call__(self, input_img, tgt, tgt_mask):
        if self.weight_type == 'onnx':
            input_name = self.nn.get_inputs()[0].name
            return self.nn.run(None, {input_name: input_img, 'tgt': tgt, 'tgt_mask': tgt_mask})[0]
        elif self.weight_type == 'trt':
            nn_input = [cupy.array(_) for _ in [input_img, tgt, tgt_mask]]
            return cupy.asnumpy(self.nn(nn_input)['output_encoder'])


class OCR(object):
    DEFAULT_TRANSFORMER_ALPHABET = "<3AT0EHFPN875RZGUBW6KSLM94DIYVOC1XJQ2>"
    DEFAULT_TRANSFORMER_PAD_SYMBOL = '#'
    alphabet = DEFAULT_TRANSFORMER_PAD_SYMBOL + DEFAULT_TRANSFORMER_ALPHABET

    def __init__(
        self, 
        weights,
        conf_thres
    ):
        self.conf = conf_thres

        # loads the model
        self.backbone = NNBackbone(weights)

        tgt = np.zeros((1, 15), dtype=int)
        tgt_mask = np.zeros((1, 15, 15), dtype=int)

        self.input_size = (150, 75)

        self.tgt = tgt
        self.tgt_mask = tgt_mask

        self.__warmup()

    def CTCGreedyDecoder(self, model_output):
        model_output = model_output.squeeze()
        classes_index = np.argmax(model_output, axis=-1)

        probs = softmax(model_output, axis=-1)
        classes_probs = np.max(probs, axis=-1)

        classes = np.array([*self.alphabet]).take(classes_index, axis=-1)

        res = []
        res_probs = []
        for element, element_prob in zip(classes, classes_probs):
            if element == '>':
                break
            if element_prob < self.conf:
                continue
            res.append(element)
            res_probs.append(element_prob)

        return res, res_probs

    def infer_plate(self, image):
        input_img = cv2.resize(image, self.input_size, interpolation=cv2.INTER_AREA)

        # preprocess input image
        input_img = input_img.astype(np.float32)
        input_img = np.transpose(input_img, (2, 0, 1))
        input_img = np.expand_dims(input_img, axis=0)

        input_img -= 127.5
        input_img /= 127.5

        # perform inference
        output = self.backbone(input_img, self.tgt, self.tgt_mask)

        self.tgt[:] = 0
        self.tgt_mask[:] = 0

        return self.CTCGreedyDecoder(output)

    def __warmup(self):
        _ = self.infer_plate(np.random.rand(*self.input_size[::-1], 3))


if __name__ == "__main__":
    images_path_list = sorted(glob.glob('images/*'))

    config = {'onnx_model_file_path': './weights/model.onnx'}

    model = OCR(config)

    for image_path in images_path_list:
        license_plate = model.infer_plate(cv2.imread(image_path))
        print(license_plate)

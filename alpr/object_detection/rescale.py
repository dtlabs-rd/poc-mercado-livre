import numpy as np

class Rescale:

    def __scale_coords(self, img1_shape, coords, img0_shape, ratio_pad=None):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            # gain  = old / new
            gain = min(img1_shape[0] / img0_shape[0],
                       img1_shape[1] / img0_shape[1])
            pad = (img1_shape[1] - img0_shape[1] * gain) / \
                2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]
        coords = np.array(coords)
        coords[[0, 2]] -= pad[0]  # x padding
        coords[[1, 3]] -= pad[1]  # y padding
        coords[:4] /= gain
        self.__clip_coords(coords, img0_shape)

        # returns the bbox position tl_x, tl_y, br_x, br_y, class, conf
        coords = np.array([int(coords[0].round()), 
                            int(coords[1].round()), 
                            int(coords[2].round()), 
                            int(coords[3].round()),
                            int(coords[4]),
                            coords[5]], dtype=np.float32)

        return coords

    def __clip_coords(self, boxes, img_shape):
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        boxes = [box if box < img_shape[1] else img_shape[1] for box in boxes]
        boxes = [box if box > 0 else 0 for box in boxes]

    def run(self, predictions, original_shape, new_shape=(640, 640)):
        coords = []
        for prediction in predictions:
            coords.append(self.__scale_coords(new_shape, prediction, original_shape))

        return np.array(coords, dtype=np.float32)
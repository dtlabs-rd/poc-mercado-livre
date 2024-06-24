import numpy as np
import time

class NonMaxSupression:

    def xywh2xyxy(self, x):
        y = np.zeros_like(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y
        
    def nms(self, dets, iou_thres):
        dets.sort(key=lambda x: x[-1], reverse=True)
        selected_dets = []
        for idx, det in enumerate(dets):
            keep = True
            for sel_det in selected_dets:
                iou = self.det_iou(det, sel_det)
                if iou > iou_thres:
                    keep = False
                    break
            if keep:
                selected_dets.append(det)

        selected_dets = [[np.array(det[0][0]), np.array(det[0][1]), np.array(
            det[1][0]), np.array(det[1][1]), det[2], det[3]] for det in selected_dets]
        return selected_dets

    def det_iou(self, d1, d2):
        tl1, tl2 = np.array(d1[0]), np.array(d2[0])
        br1, br2 = np.array(d1[1]), np.array(d2[1])

        a1 = np.prod(br1 - tl1)
        a2 = np.prod(br2 - tl2)
        tl = np.maximum(tl1, tl2)
        br = np.minimum(br1, br2)
        intersec = np.prod(np.maximum(br-tl, 0.))
        union = a1+a2-intersec
        return intersec/union
        
    def run(self, prediction,  conf_thres=0.3, iou_thres=0.6):
        filtered_dets = [None]
        nc = prediction[0].shape[1] - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates
        # Settings
        # (pixels) minimum and maximum box width and height
        min_wh, max_wh = 2, 4096
        max_det = 300  # maximum number of detections per image
        time_limit = 10.0  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

        t = time.time()
        output = [None] * prediction.shape[0]
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence
            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = self.xywh2xyxy(x[:, :4])

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = np.transpose((x[:, 5:] > conf_thres).nonzero())[:, 0], np.transpose((x[:, 5:] > conf_thres).nonzero())[:, 1]
                x = np.concatenate((box[i], x[i, j + 5, None], j[:, None]), 1)
            else:  # best class only
                j = np.array(
                    [[float(np.where(value == np.amax(value))[0]) for value in x[:, 5:]]])[0]
                j = np.expand_dims(j, axis=1)
                conf = np.max(x[:, 5:], axis=1, keepdims=True)
                x = np.concatenate((box, conf, j), 1)[conf[:, 0] > conf_thres]

            # If none remain process next image
            n = x.shape[0]  # number of boxes
            if not n:
                continue

            # Batched NMS
            # boxes (offset by class), scores
            boxes, scores, class_ = x[:, :4], x[:, 4], x[:, -1]
            dets = [[[box[0], box[1]], [box[2], box[3]], class_[
                idx], scores[idx]] for idx, box in enumerate(boxes)]
            filtered_dets = self.nms(dets, iou_thres)
            if len(filtered_dets) > max_det:  # limit detections
                filtered_dets = filtered_dets[:max_det]
            if (time.time() - t) > time_limit:
                break  # time limit exceeded
            output[xi] = filtered_dets

        return output
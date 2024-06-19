from ALPR import ALPR   
import cv2

alpr_pipeline = ALPR(
    det_weights='../models/alpr_detector.onnx',
    ocr_weights='../models/alpr_ocr.onnx'
)

img = alpr_pipeline.prepare_image('../data/test_alpr.jpg')
results = alpr_pipeline(img)

for result in results:
    x1, y1, x2, y2 = map(int, result['detection'][:4])
    cv2.rectangle(img, (x1, y1), (x2, y2), color=(0,0,255))
    cv2.putText(img, result['text'], (x1, y1), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0,0,255), thickness=1)

cv2.imshow('Detections', img)
cv2.waitKey(0)
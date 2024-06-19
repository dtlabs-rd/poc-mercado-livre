from ALPR import ALPR   
import cv2
import time

alpr_pipeline = ALPR(
    det_weights='../models/alpr_detector.onnx',
    ocr_weights='../models/alpr_ocr.onnx'
)

start = time.time()
img = alpr_pipeline.prepare_image('../data/test_alpr.jpg')
results = alpr_pipeline(img)
end = time.time()

print(f"Inference took {(end-start)} seconds.")

for result in results:
    x1, y1, x2, y2 = map(int, result['detection'][:4])
    cv2.rectangle(img, (x1, y1), (x2, y2), color=(0,0,255))
    cv2.putText(img, result['text'], (x1, y1), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0,0,255), thickness=1)

cv2.imshow('Results', img)
cv2.waitKey(0)
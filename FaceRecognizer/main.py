from FaceRecognizer import FaceRecognizer
import cv2
import time
        
# Face recognition pipeline
face_recognition_pipeline = FaceRecognizer(use_colors=True)

# Inference
start = time.time()
img = face_recognition_pipeline.prepare_image('../data/test_face.jpg')
results = face_recognition_pipeline(img)
end = time.time()

print(f"Inference took {(end-start)} seconds.")

# Results
for result in results:
    print("Embedding shape: ", result.vector().shape)
    x1, y1, x2, y2 = map(int, result.face.bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), color=(0,0,255))

cv2.imshow('Results', img)
cv2.waitKey(0)
    
    
    
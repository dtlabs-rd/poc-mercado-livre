from FaceRecognizer import FaceRecognizer
        
# Face recognition pipeline
pipeline = FaceRecognizer(use_colors=True)

# Inference
faces = pipeline('../data/test_face_recognition.jpg')

# Results
for face in faces:
    print("Embedding: ", face.vector())
    print("Direction: ", face.direction())
    print("Angle: ", face.angle())
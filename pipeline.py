import cv2
from alpr.alpr import ALPR
from face_recognizer.face_recognizer import FaceRecognizer
from database.database import Database
import base64
import json
import time

# Thresholds
PLATE_DISTANCE_THRESHOLD = 2
FACE_SIMILARITY_THRESHOLD = 0.5

# Streams
alpr_stream = cv2.VideoCapture("./data/video_0_94.mp4")
face_recognition_stream = cv2.VideoCapture(1)

# Database
database = Database("./database.npz")

# ALPR pipeline
alpr_pipeline = ALPR(
    det_weights="./models/alpr_detector.onnx",
    ocr_weights="./models/alpr_ocr.onnx"
)

# Face Recognition pipeline
face_recognition_pipeline = FaceRecognizer()

while True:
    
    start = time.time()
    results = {}
    
    # Read frame from streams
    timestamp = time.time()
    _, alpr_frame = alpr_stream.read()
    _, face_recognition_frame = face_recognition_stream.read()
        
    # Run ALPR pipeline
    alpr_result = alpr_pipeline(alpr_frame)
    
    if len(alpr_result) > 0:
        
        # Search for plate in database
        plate, levenshtein_distance = database.query_plate(
            query_plate=alpr_result[0]['text'], 
            top_k=1
        )[0]
        
        # Apply plate levenshtein distance threshold
        if levenshtein_distance < PLATE_DISTANCE_THRESHOLD:
            
            results["plate_text"] = plate
            results["plate_bbox"] = alpr_result[0]['detection'].tolist()
            
            # Run Face Recognition pipeline
            face_recognition_result = face_recognition_pipeline(face_recognition_frame)
            
            if len(face_recognition_result) > 0:
                
                # Search for face in database
                person, similarity = database.query_face(
                    query_embedding=face_recognition_result[0].embedding(),
                    top_k=1
                )[0]
            
                # Apply face similarity threshold
                if similarity > FACE_SIMILARITY_THRESHOLD:
                    results["valid"] = plate in person['plates']
                    results["name"] = person['name']
                    results["face_bbox"] = face_recognition_result[0].face['bbox'].tolist()
                    results["face_kpts"] = face_recognition_result[0].face['kps'].tolist()
            
    results["alpr_frame"] =  base64.b64encode(
        cv2.imencode('.jpg', alpr_frame)[1].tobytes()
    ).decode('utf-8')
    results["face_recognition_frame"] = base64.b64encode(
        cv2.imencode('.jpg', face_recognition_frame)[1].tobytes()
    ).decode('utf-8')
        
    with open(f"./results/{int(timestamp*1000)}.json", "w") as fp:
        json.dump(results, fp)
        
    end = time.time()
    time.sleep(max((1/30) - (end-start), 0))

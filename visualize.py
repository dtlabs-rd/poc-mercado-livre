import cv2
import os
import glob
import json
import numpy as np
import base64

RESULTS_FOLDER = "./results/"

def readb64(data):
   nparr = np.fromstring(base64.b64decode(data), np.uint8)
   img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   return img

while True:
    
    files = glob.glob(os.path.join(RESULTS_FOLDER, '*'))
    files.sort()
    
    for file in files:
    
        with open(file, 'r') as fp:
            data = json.load(fp)
        
        # Load ALPR data
        alpr_frame = readb64(data['alpr_frame'])
        plate_bbox = data['plate_bbox']
        plate_text = data['plate_text']
        
        # Draw ALPR results
        x1, y1, x2, y2 = map(int, plate_bbox[:4])
        cv2.rectangle(alpr_frame, (x1, y1), (x2, y2), color=(0,255,0), thickness=2)
        cv2.putText(alpr_frame, plate_text, (x1, y1), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,255,0), thickness=2)

        # Load Face Recognition results
        face_recognition_frame = readb64(data['face_recognition_frame'])
        face_bbox = data['face_bbox']
        face_kpts = data['face_kpts']
        
        # Draw Face Recognition results
        x1, y1, x2, y2 = map(int, face_bbox[:4])
        cv2.rectangle(face_recognition_frame, (x1, y1), (x2, y2), (0,255,0))
        for kpt in face_kpts[:2]:
            x, y = map(int, kpt)
            cv2.circle(face_recognition_frame, (x, y), 2, (0,255,0), thickness=2)
            
        h1, w1 = face_recognition_frame.shape[:2]
        h2, w2 = alpr_frame.shape[:2]
        vis = np.zeros((max(h1, h2), w1+w2,3), np.uint8)
        vis[:h1, :w1,:3] = face_recognition_frame
        vis[:h2, w1:w1+w2,:3] = alpr_frame

        cv2.imshow("vis", vis)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
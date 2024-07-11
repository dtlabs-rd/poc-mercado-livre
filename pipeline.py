import cv2
from alpr.alpr import ALPR
from rabbitmq import RabbitMQ, Config
import base64
import json
import time
from collections import deque
from threading import Thread, Lock
import boto3
from dotenv import load_dotenv
load_dotenv()

# RabbitMQ settings
RABBITMQ_EXCHANGE = "poc-meli"
RABBITMQ_QUEUE = "poc-meli"
rabbitmq = RabbitMQ(Config(), RABBITMQ_EXCHANGE, "direct")
channel = rabbitmq.get_channel()
arguments = {
    "x-queue-type": "stream", 
    "x-max-age": "1m", 
    "x-stream-max-segment-size-bytes":1_000_000
}
result = channel.queue_declare(
    queue=RABBITMQ_QUEUE, 
    durable=True, 
    arguments=arguments, 
    auto_delete=False
)
channel.queue_bind(
    exchange=RABBITMQ_EXCHANGE, 
    queue=result.method.queue,
    routing_key=RABBITMQ_QUEUE
)

# S3 settings
s3_client = boto3.client('s3')

# Streams
alpr_frame_lock = Lock()
alpr_frame = None
def read_alpr_stream():
    global alpr_frame
    alpr_stream = cv2.VideoCapture("./data/output_2024-07-10_11-27-07.mp4")
    while True:
        time.sleep(1/10)
        _, frame = alpr_stream.read()
        if frame is not None:
            with alpr_frame_lock:
                alpr_frame = frame
        
read_stream_thread = Thread(target=read_alpr_stream)
read_stream_thread.daemon = True
read_stream_thread.start()

# ALPR pipeline
alpr_pipeline = ALPR(
    det_weights="./models/alpr_detector.onnx",
    ocr_weights="./models/alpr_ocr.onnx",
    use_tracker=True
)

# Queue to store last plates
queue = deque(maxlen=15)
last_id = -1
temp_id = -1

while True:
    start = time.time()
    results = {}
    
    # Get frames
    with alpr_frame_lock:
        if alpr_frame is not None:
            alpr_frame_copy = alpr_frame.copy()
        else:
            continue
        
    # Resize frame by scale_factor
    scale_factor = 0.5
    alpr_frame_copy = cv2.resize(alpr_frame_copy, (0,0), fx=scale_factor, fy=scale_factor)
        
    # Run ALPR pipeline
    alpr_results = alpr_pipeline(alpr_frame)
    
    if len(alpr_results) > 0:
        alpr_frame_encoded =  base64.b64encode(
            cv2.imencode('.jpg', alpr_frame_copy)[1].tobytes()
        ).decode('utf-8')
    
    for alpr_result in alpr_results:   
        
        if len(alpr_result['text']) != 7:
            continue
        
        if alpr_result['id'] > last_id:
            temp_id = max(temp_id, alpr_result['id'])
            queue.append({
                "plate": alpr_result['text'],
                "timestamp": int(start*1000)
            })
        
        # Draw ALPR results
        x1, y1, x2, y2 = map(lambda x: int(x*scale_factor), alpr_result['detection'].tolist()[:4])
        cv2.rectangle(
            alpr_frame_copy, 
            (x1, y1), 
            (x2, y2), 
            color=(0,255,0), 
            thickness=2
        )
        cv2.putText(
            alpr_frame_copy, 
            f'ID: {alpr_result["id"]} - {alpr_result["text"]}', 
            (x1, y1-5), 
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=1, 
            color=(0,255,0), 
            thickness=2
        )
        
        # Save in S3 Bucket
        s3_json = {
            "plate": alpr_result["text"],
            "image": alpr_frame_encoded
        }
        s3_client.put_object(
            Key=f"{int(time.time()*1000)}.json",
            Body=json.dumps(s3_json),
            Bucket='poc-meli'
        )
            
    results["alpr_frame"] =  base64.b64encode(
        cv2.imencode('.jpg', alpr_frame_copy)[1].tobytes()
    ).decode('utf-8')
    
    last_id = temp_id
    results["history"] = list(queue)
    
    # Publish to RABBITMQ
    channel = rabbitmq.get_channel()
    channel.basic_publish(
        RABBITMQ_EXCHANGE,
        routing_key=RABBITMQ_QUEUE,
        body=json.dumps(results)
    ) 
        
    end = time.time()
    time.sleep(max((1/10) - (end-start), 0))

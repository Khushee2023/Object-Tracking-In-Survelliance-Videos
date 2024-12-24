import os
import cv2
import math
import json
import base64
import time
from flask import Blueprint, render_template, request, Response, send_file, jsonify
from werkzeug.utils import secure_filename
import numpy as np

object_tracking_bp = Blueprint('object_tracking', __name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
MODEL_PATH = os.path.join('models')

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Detection parameters
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

class ObjectTracker:
    def __init__(self):
        self.tracking_objects = {}
        self.track_id = 0
        
    def update_tracks(self, center_points):
        new_tracking_objects = {}
        
        for pt in center_points:
            same_object_detected = False
            
            for object_id, prev_pt in self.tracking_objects.items():
                distance = math.hypot(prev_pt[0] - pt[0], prev_pt[1] - pt[1])
                
                if distance < 35:  # Distance threshold for same object
                    new_tracking_objects[object_id] = pt
                    same_object_detected = True
                    break
            
            if not same_object_detected:
                new_tracking_objects[self.track_id] = pt
                self.track_id += 1
        
        self.tracking_objects = new_tracking_objects
        return self.tracking_objects

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    # Load YOLO model
    config_path = os.path.join(MODEL_PATH, 'yolov4-tiny.cfg')
    weights_path = os.path.join(MODEL_PATH, 'yolov4-tiny.weights')
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    
    # Load class names
    classes_path = os.path.join(MODEL_PATH, 'coco.names')
    with open(classes_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    return net, classes

def detect_objects(frame, net, classes):
    """Detect objects in a frame using YOLOv4-tiny."""
    height, width = frame.shape[:2]
    
    # Create blob from image
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    
    # Get output layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    # Forward pass
    detections = net.forward(output_layers)
    
    # Initialize lists for detected objects
    boxes = []
    class_ids = []
    confidences = []
    center_points = []
    
    # Process detections
    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > CONF_THRESHOLD:
                # Scale coordinates to original image size
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                
                boxes.append([x, y, w, h])
                class_ids.append(class_id)
                confidences.append(float(confidence))
                center_points.append((center_x, center_y))
    
    # Apply Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
    
    result_objects = []
    result_center_points = []
    
    if len(indices) > 0:
        for i in indices.flatten():
            result_objects.append({
                'box': boxes[i],
                'class': classes[class_ids[i]],
                'confidence': confidences[i]
            })
            result_center_points.append(center_points[i])
    
    return result_objects, result_center_points

@object_tracking_bp.route('/')
def index():
    return render_template('object_tracking.html')

@object_tracking_bp.route('/process_video', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return 'No video file provided', 400
    
    video_file = request.files['video']
    if video_file.filename == '' or not allowed_file(video_file.filename):
        return 'Invalid file type', 400

    # Save uploaded video
    filename = secure_filename(video_file.filename)
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    video_file.save(video_path)

    def generate():
        # Initialize model and tracker
        net, classes = load_model()
        tracker = ObjectTracker()
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        processing_start = cv2.getTickCount()
        
        # Initialize video writer for saving the processed video
        output_path = os.path.join(UPLOAD_FOLDER, f'processed_{filename}')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = None
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Resize for faster processing
                frame = cv2.resize(frame, (640, 360))  # Resize to 640x360 for streaming
                objects, center_points = detect_objects(frame, net, classes)
                
                # Update object tracking
                tracker.update_tracks(center_points)
                
                # Draw bounding boxes and track IDs
                for obj in objects:
                    x, y, w, h = obj['box']
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    label = f"{obj['class']} {obj['confidence']:.2f}"
                    cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Encode frame to JPEG
                _, jpeg_frame = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                jpeg_data = base64.b64encode(jpeg_frame).decode('utf-8')

                # Send data over SSE
                yield f"data: {json.dumps({'image': 'data:image/jpeg;base64,' + jpeg_data})}\n\n"
                
                # Add FPS calculation
                if frame_count % 60 == 0:
                    current_time = cv2.getTickCount()
                    time_diff = (current_time - processing_start) / cv2.getTickFrequency()
                    fps = frame_count / time_diff
                    yield f"data: {json.dumps({'fps': fps})}\n\n"
        
        finally:
            cap.release()

    return Response(generate(), content_type='text/event-stream')

@object_tracking_bp.route('/download_result')
def download_result():
    processed_video_path = os.path.join(UPLOAD_FOLDER, 'processed_video.avi')
    return send_file(processed_video_path, as_attachment=True)

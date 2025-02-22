from flask import Blueprint, request, jsonify, render_template, send_from_directory
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import uuid
import threading
import math

object_tracking_bp = Blueprint("object_tracking", __name__)

# Constants and directory setup
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load YOLO model and classes
net = cv2.dnn.readNetFromDarknet(
    os.path.join("yolov4-tiny.cfg"),
    os.path.join("yolov4-tiny.weights")
)

with open(os.path.join("classes.txt"), "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Store task information
tasks = {}

def detect_objects(frame, conf_threshold, nms_threshold):
    """Detect objects in a frame using YOLO."""
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > conf_threshold:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
    return ([boxes[i] for i in indices], 
            [class_ids[i] for i in indices],
            [confidences[i] for i in indices]) if len(indices) > 0 else ([], [], [])

def process_video(task_id, input_path, output_path, conf_threshold, nms_threshold):
    """Process video with object detection and tracking."""
    try:
        cap = cv2.VideoCapture(input_path)
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize video writer
        output_video = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (frame_width, frame_height)
        )

# Initialize tracking variables
        tracking_objects = {}
        track_id = 0
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            center_points_cur_frame = []

            # Detect objects
            boxes, class_ids, confidences = detect_objects(frame, conf_threshold, nms_threshold)
            
            # Process detections
            for i, box in enumerate(boxes):
                x, y, w, h = box
                cx = int((x + x + w) / 2)
                cy = int((y + y + h) / 2)
                center_points_cur_frame.append((cx, cy))
 # Draw detection box and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"{class_names[class_ids[i]]}: {confidences[i]:.2f}"
                cv2.putText(frame, label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Update tracking
            new_tracking_objects = {}
            for pt in center_points_cur_frame:
                same_object_detected = False
                for obj_id, prev_pt in tracking_objects.items():
                    distance = math.hypot(prev_pt[0] - pt[0], prev_pt[1] - pt[1])
                    if distance < 35:  # Tracking threshold
                        new_tracking_objects[obj_id] = pt
                        same_object_detected = True
                        break

                if not same_object_detected:
                    new_tracking_objects[track_id] = pt
                    track_id += 1

            tracking_objects = new_tracking_objects

 # Draw tracking IDs
            for obj_id, pt in tracking_objects.items():
                cv2.circle(frame, pt, 5, (0, 0, 255), -1)
                cv2.putText(frame, str(obj_id), (pt[0] - 10, pt[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Write frame
            output_video.write(frame)
            
            # Update progress
            progress = int((frame_count / total_frames) * 100)
            tasks[task_id]['progress'] = progress

        cap.release()
        output_video.release()
        tasks[task_id]['complete'] = True
        
    except Exception as e:
        tasks[task_id]['error'] = str(e)
        raise
@object_tracking_bp.route('/')
def index():
    """Render the object tracking page."""
    return render_template('object_tracking.html')

@object_tracking_bp.route('/upload', methods=['POST'])
def upload_video():
    """Handle video upload and start processing."""
    if 'videoFile' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'})

    video_file = request.files['videoFile']
    if video_file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'})

    try:
        # Get parameters
        conf_threshold = float(request.form.get('confidenceThreshold', 0.5))
        nms_threshold = float(request.form.get('nmsThreshold', 0.4))
# Save video
        task_id = str(uuid.uuid4())
        video_path = os.path.join(UPLOAD_FOLDER, f'{task_id}.mp4')
        output_path = os.path.join(OUTPUT_FOLDER, f'{task_id}_processed.mp4')
        video_file.save(video_path)
        
        # Initialize task
        tasks[task_id] = {
            'video_path': video_path,
            'output_path': output_path,
            'progress': 0,
            'complete': False,
            'error': None
        }

        # Start processing thread
        threading.Thread(
            target=process_video,
            args=(task_id, video_path, output_path, conf_threshold, nms_threshold)
        ).start()
 return jsonify({'success': True, 'taskId': task_id})

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@object_tracking_bp.route('/status/<task_id>')
def task_status(task_id):
    """Get the status of a processing task."""
    task = tasks.get(task_id)
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    
    return jsonify({
        'progress': task['progress'],
        'complete': task['complete'],
        'error': task['error'],
        'outputFile': f'{task_id}_processed.mp4' if task['complete'] else None
    })

@object_tracking_bp.route('/output/<filename>')
def output_file(filename):
    """Serve processed video file."""
    return send_from_directory(OUTPUT_FOLDER, filename)

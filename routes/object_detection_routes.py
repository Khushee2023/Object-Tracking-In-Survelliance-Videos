from flask import Blueprint, request, jsonify, render_template, send_from_directory
import os
from werkzeug.utils import secure_filename
import uuid
import threading
from imageai.Detection import VideoObjectDetection
import cv2

object_detection_bp = Blueprint("object_detection", __name__)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

tasks = {}

def process_video(task_id, input_path, output_path, confidence):
    try:
        # Calculate total frames using OpenCV
        cap = cv2.VideoCapture(input_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        video_detector = VideoObjectDetection()
        video_detector.setModelTypeAsTinyYOLOv3()
        video_detector.setModelPath("models/tiny-yolov3.pt")
        video_detector.loadModel()

        def per_frame_function(frame_number, output_array, output_count):
            progress = int((frame_number / total_frames) * 100)
            tasks[task_id]["progress"] = progress

        video_detector.detectObjectsFromVideo(
            input_file_path=input_path,
            output_file_path=output_path,
            frames_per_second=10,
            per_frame_function=per_frame_function,
            minimum_percentage_probability=confidence
        )
        
        tasks[task_id]["complete"] = True
        tasks[task_id]["outputFile"] = os.path.basename(output_path) + ".mp4"
    except Exception as e:
        tasks[task_id]["error"] = str(e)
        tasks[task_id]["complete"] = True

@object_detection_bp.route('/')
def index():
    return render_template('object_detection.html')

@object_detection_bp.route("/upload", methods=["POST"])
def upload_video():
    if "videoFile" not in request.files:
        return jsonify({"success": False, "message": "No file uploaded"})

    video_file = request.files["videoFile"]
    if video_file.filename == "":
        return jsonify({"success": False, "message": "No selected file"})

    confidence = int(request.form.get("confidenceThreshold", 30))
    
    task_id = str(uuid.uuid4())
    input_filename = secure_filename(video_file.filename)
    output_filename = f"detected_{task_id}"
    
    input_path = os.path.join(UPLOAD_FOLDER, input_filename)
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
    
    video_file.save(input_path)
    tasks[task_id] = {"complete": False, "progress": 0}

    thread = threading.Thread(
        target=process_video,
        args=(task_id, input_path, output_path, confidence)
    )
    thread.daemon = True
    thread.start()

    return jsonify({"success": True, "taskId": task_id})

@object_detection_bp.route("/status/<task_id>")
def get_status(task_id):
    if task_id not in tasks:
        return jsonify({"success": False, "message": "Invalid task ID"})
    return jsonify(tasks[task_id])

@object_detection_bp.route("/output/<filename>")
def get_output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

from flask import Blueprint, request, jsonify, render_template, send_from_directory
import os
from werkzeug.utils import secure_filename
import uuid
import threading
import cv2
import numpy as np

dense_opt_flow_bp = Blueprint("dense_opt_flow", __name__)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

tasks = {}

def process_video(task_id, input_path, output_path):
    try:
        cap = cv2.VideoCapture(input_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        
        ret, first_frame = cap.read()
        prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mask = np.zeros_like(frame)
            mask[..., 1] = 255
            mask[..., 0] = angle * 180 / np.pi / 2
            mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

            out.write(rgb)

            prev_gray = gray

            tasks[task_id]['progress'] = int((cap.get(cv2.CAP_PROP_POS_FRAMES) / total_frames) * 100)
            
        cap.release()
        out.release()
        tasks[task_id]['complete'] = True
    except Exception as e:
        tasks[task_id]['error'] = str(e)

@dense_opt_flow_bp.route('/')
def index():
    return render_template('dense_opt_flow.html')

@dense_opt_flow_bp.route('/upload', methods=['POST'])
def upload_video():
    if 'videoFile' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'})

    video_file = request.files['videoFile']
    if video_file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'})

    filename = secure_filename(video_file.filename)
    task_id = str(uuid.uuid4())
    video_path = os.path.join(UPLOAD_FOLDER, f'{task_id}.mp4')
    video_file.save(video_path)
    
    output_path = os.path.join(OUTPUT_FOLDER, f'{task_id}_processed.mp4')
    
    tasks[task_id] = {
        'video_path': video_path,
        'output_path': output_path,
        'progress': 0,
        'complete': False,
        'error': None
    }

    threading.Thread(target=process_video, args=(task_id, video_path, output_path)).start()

    return jsonify({'success': True, 'taskId': task_id})

@dense_opt_flow_bp.route('/status/<task_id>')
def task_status(task_id):
    task = tasks.get(task_id)
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    return jsonify({'progress': task['progress'], 'complete': task['complete'], 'outputFile': f'{task_id}_processed.mp4', 'error': task.get('error')})

@dense_opt_flow_bp.route('/output/<filename>')
def output_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

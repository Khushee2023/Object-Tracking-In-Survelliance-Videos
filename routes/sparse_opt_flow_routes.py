from flask import Blueprint, request, jsonify, render_template, send_from_directory
import os
from werkzeug.utils import secure_filename
import uuid
import threading
import cv2
import numpy as np

sparse_opt_flow_bp = Blueprint("sparse_opt_flow", __name__)

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
        
        lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        
        ret, old_frame = cap.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        
        mask = np.zeros_like(old_frame)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
            
            img = cv2.add(frame, mask)
            out.write(img)
            
            p0 = good_new.reshape(-1, 1, 2)
            old_gray = frame_gray.copy()

            tasks[task_id]['progress'] = int((cap.get(cv2.CAP_PROP_POS_FRAMES) / total_frames) * 100)
            
        cap.release()
        out.release()
        tasks[task_id]['complete'] = True
    except Exception as e:
        tasks[task_id]['error'] = str(e)

@sparse_opt_flow_bp.route('/')
def index():
    return render_template('sparse_opt_flow.html')

@sparse_opt_flow_bp.route('/upload', methods=['POST'])
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

@sparse_opt_flow_bp.route('/status/<task_id>')
def task_status(task_id):
    task = tasks.get(task_id)
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    return jsonify({'progress': task['progress'], 'complete': task['complete'], 'outputFile': f'{task_id}_processed.mp4', 'error': task.get('error')})

@sparse_opt_flow_bp.route('/output/<filename>')
def output_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

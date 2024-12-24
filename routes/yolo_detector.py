import cv2
import numpy as np
import os

class YOLODetector:
    def __init__(self):
        base_path = os.path.dirname(os.path.dirname(__file__))
        models_path = os.path.join(base_path, 'models')
        
        self.config_path = os.path.join(models_path, "yolov3-tiny.cfg")
        self.weights_path = os.path.join(models_path, "yolov3-tiny.weights")
        self.coco_path = os.path.join(models_path, "coco.names")
        
        # Initialize variables
        self.current_task_id = None
        
        # Load YOLO
        self.classes = []
        with open(self.coco_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        self.net = cv2.dnn.readNetFromDarknet(self.config_path, self.weights_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def detect_frame(self, frame, conf_threshold):
        height, width = frame.shape[:2]
        
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        
        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > conf_threshold/100:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold/100, 0.4)
        
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = self.classes[class_ids[i]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {confidences[i]:.2f}', (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame

    def process_video(self, input_path, output_path, confidence):
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            raise Exception("Error opening video file")
            
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        processed_frames = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            processed_frame = self.detect_frame(frame, confidence)
            out.write(processed_frame)
            
            processed_frames += 1
            if self.current_task_id:
                from . import object_detection_routes
                object_detection_routes.tasks[self.current_task_id]["progress"] = \
                    int((processed_frames / total_frames) * 100)
            
        cap.release()
        out.release()
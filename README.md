# KhusheeRanjan_Object_Detection
# Image and Video Processing Application

This application provides a set of routes for image processing, object detection, and object tracking using Flask. It utilizes OpenCV and ImageAI for various image and video manipulation tasks.

## Features

### Image Processing
- Generate histograms of images.
- Split images into RGB channels.
- Apply Gaussian blur to images.

### Object Detection
- Detect objects in videos using the YOLO (You Only Look Once) model.
- Track progress of video processing tasks.

### Object Tracking
- Track objects across video frames using a custom object tracker.

## Requirements
- Python 3.x
- Flask
- OpenCV
- NumPy
- ImageAI
- Other dependencies as specified in `requirements.txt`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Khushee2023/KhusheeRanjan_Object_Detection
   cd https://github.com/Khushee2023/KhusheeRanjan_Object_Detection

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   
3. Download the YOLO model files and place them in the models directory:
   * yolov3-tiny.cfg
   * yolov3-tiny.weights
   * coco.names
  
## Directory Structure

MultipleFiles/
├── models/                     # Directory for model files
│   ├── yolov3-tiny.cfg
│   ├── yolov3-tiny.weights
│   └── coco.names
├── uploads/                    # Directory for uploaded files
├── processed/                  # Directory for processed files
├── image_processing_routes.py  # Image processing routes
├── object_detection_routes.py  # Object detection routes
├── object_tracking_routes.py   # Object tracking routes
└── yolo_detector.py            # YOLO detection logic


## Usage

1. Start the Flask application:
   ```bash
   flask run
2. Access the application in your web browser at http://127.0.0.1:5000.
3. Use the following endpoints:

## API Endpoints

### Image Processing
* GET /: Render the image processing page.
* POST /process_image: Process the uploaded image based on the selected action.

### Object Detection

* GET /: Render the object detection page.
* POST /upload: Upload a video for object detection.
* GET /status/<task_id>: Get the status of the video processing task.
* GET /output/<filename>: Download the processed video.

### Object Tracking

* GET /: Render the object tracking page.
* POST /process_video: Upload a video for object tracking.
* GET /download_result: Download the processed tracking video.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

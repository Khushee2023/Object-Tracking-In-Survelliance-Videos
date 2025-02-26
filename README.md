# Vision-X 🚀  
*A Flask-based Computer Vision Web App*  

![Vision-X](https://your-image-link.com) <!-- (Optional: Add a project banner) -->  

## 🌟 Overview  
Vision-X is a powerful web-based computer vision application integrating multiple deep learning models for image processing, object detection, tracking, and optical flow analysis. Built with Flask, it provides an intuitive UI for seamless interaction.  

## 🛠️ Features  
- 🎨 **Image Processing** - Various transformations and filters  
- 🎯 **Object Detection** - YOLOv3-based detection  
- 🚶 **Object Tracking** - YOLOv4 Tiny for real-time tracking  
- 🔍 **Sparse Optical Flow** - Motion tracking with Lucas-Kanade method  
- 🌊 **Dense Optical Flow** - Farneback-based flow visualization  

## 💻 Tech Stack  
- **Frontend**: HTML, CSS, JavaScript  
- **Backend**: Flask (Python)  
- **Deep Learning**: OpenCV, PyTorch, YOLO models  
- **Deployment**: 

## 📂 Project Structure  
```bash
Vision-X/
│── app/
│   ├── static/           # CSS, JS, images
│   ├── templates/        # HTML templates
│── routes/
│   ├── image_processing_routes.py
│   ├── object_detection_routes.py
│   ├── object_tracking_routes.py
│   ├── sparse_opt_flow_routes.py
│   ├── dense_opt_flow_routes.py
│── models/               # YOLO model weights & config
│── uploads/              # Uploaded videos
│── processed/            # Processed outputs
│── app.py                # Main Flask app
│── requirements.txt      # Dependencies
│── README.md             # Project documentation


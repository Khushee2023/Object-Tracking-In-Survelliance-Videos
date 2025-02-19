import os
import requests
from flask import Flask, render_template
from flask_sock import Sock
from flask_cors import CORS
from routes.image_processing_routes import image_processing_bp
from routes.object_detection_routes import object_detection_bp
from routes.object_tracking_routes import object_tracking_bp
from routes.sparse_opt_flow_routes import sparse_opt_flow_bp
from routes.dense_opt_flow_routes import dense_opt_flow_bp

app = Flask(__name__, template_folder='app/templates', static_folder='app/static')

sock = Sock(app)
CORS(app)

app.register_blueprint(image_processing_bp, url_prefix='/model1')
app.register_blueprint(object_detection_bp, url_prefix='/model2')
app.register_blueprint(object_tracking_bp, url_prefix='/model3')
app.register_blueprint(sparse_opt_flow_bp, url_prefix='/model4')
app.register_blueprint(dense_opt_flow_bp, url_prefix='/model5')

# Folder where models will be stored
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Dictionary of required model files and their Google Drive direct links
model_files = {
    "yolov3.weights": "https://drive.google.com/uc?export=download&id=1jIc6Bjs3cqqhZxcFrVb57c3ZktKqzT7K",
    "yolov3-tiny.weights": "https://drive.google.com/uc?export=download&id=1odWA02_wfKLKpZsJJ0fv4nTkdlBWHhQ7",
    "model.pt": "https://drive.google.com/uc?export=download&id=1KQaFg0rvHKpz410a3qPkPD_KhhCZe1Nu"
}

# Function to download missing model files
def download_models():
    for file_name, url in model_files.items():
        file_path = os.path.join(MODEL_DIR, file_name)
        if not os.path.exists(file_path):
            print(f"Downloading {file_name}...")
            response = requests.get(url, stream=True)
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"{file_name} downloaded successfully!")

# Call the function before running the app
download_models()

@app.route('/')
def onboarding():
    return render_template('onboarding.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)

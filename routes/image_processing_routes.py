from flask import Blueprint, render_template, request, send_file
import cv2
import numpy as np
from io import BytesIO

# Blueprint declaration
image_processing_bp = Blueprint(
    'image_processing', __name__, 
    template_folder='../../app/templates',  # Correct relative path
    static_folder='../../app/static'        # Correct static path
)

# Route to render the image processing page
@image_processing_bp.route('/', methods=['GET'])
def image_processing_page():
    return render_template('image_processing.html')

# Route to process image actions
@image_processing_bp.route('/process_image', methods=['POST']) 
def process_image():
    action = request.form.get('action')
    file = request.files['file']

    if not file:
        return 'No file uploaded', 400

    # Read the uploaded image
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    if action == 'histogram':
        # Convert image to grayscale for histogram
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        # Create a blank image and draw the histogram
        hist_img = np.zeros((400, 256, 3), dtype=np.uint8)
        cv2.normalize(hist, hist, 0, 400, cv2.NORM_MINMAX)
        
        for x, h in enumerate(hist):
            cv2.line(hist_img, (x, 400), (x, 400 - int(h)), (255, 255, 255), 1)
        
        # Send histogram image
        _, buffer = cv2.imencode('.jpg', hist_img)
        return send_file(BytesIO(buffer), mimetype='image/jpeg')

    elif action == 'rgb_channels':
        # Split image into BGR channels
        b, g, r = cv2.split(image)

        # Create images for each channel
        r_img = cv2.merge([r, np.zeros_like(g), np.zeros_like(b)])
        g_img = cv2.merge([np.zeros_like(r), g, np.zeros_like(b)])
        b_img = cv2.merge([np.zeros_like(r), np.zeros_like(g), b])

        # Stack the channel images vertically
        rgb_img = np.vstack([r_img, g_img, b_img])

        # Send stacked RGB channels image
        _, buffer = cv2.imencode('.jpg', rgb_img)
        return send_file(BytesIO(buffer), mimetype='image/jpeg')

    elif action == 'blur':
        # Apply Gaussian Blur
        image = cv2.GaussianBlur(image, (15, 15), 0)

        # Encode and send processed image
        _, buffer = cv2.imencode('.jpg', image)
        return send_file(BytesIO(buffer), mimetype='image/jpeg')

    # In case of unknown action
    return 'Unknown action', 400


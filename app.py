import os
from flask import Flask, render_template
from routes.object_tracking_routes import object_tracking_bp  # Model 3
from routes.image_processing_routes import image_processing_bp  # Model 1
from routes.object_detection_routes import object_detection_bp  # Model 2
import warnings

app = Flask(
    __name__,
    template_folder='app/templates',
    static_folder='app/static'
)

warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*")

# Register blueprints for all models
app.register_blueprint(image_processing_bp, url_prefix='/model1')
app.register_blueprint(object_detection_bp, url_prefix='/model2')
app.register_blueprint(object_tracking_bp, url_prefix='/model3')

@app.route('/')
def onboarding():
    return render_template('onboarding.html')

if __name__ == '__main__':
    app.run(debug=True)





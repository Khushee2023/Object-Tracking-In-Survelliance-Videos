import os
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

@app.route('/')
def onboarding():
    return render_template('onboarding.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Default port is 10000 if PORT is not set
    app.run(host='0.0.0.0', port=port, debug=True)

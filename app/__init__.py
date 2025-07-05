from flask import Flask, send_from_directory
from flask_cors import CORS
import os

def create_app():
    app = Flask(__name__, static_folder='../static', static_url_path='/')
    
    # CORS(app)
    # Allow any origin (for testing only!)
    CORS(app, resources={r"/predict": {"origins": "*"}})

    # Register Blueprint predict
    from .routes.predict import predict_bp
    app.register_blueprint(predict_bp)

    # Route utama untuk serve index.html
    @app.route('/')
    def serve_index():
        return send_from_directory(app.static_folder, 'index.html')

    # Optional: Fallback route untuk SPA (misalnya kalau pakai routing frontend)
    @app.errorhandler(404)
    def not_found(e):
        return send_from_directory(app.static_folder, 'index.html')

    return app

from flask import Blueprint, request, jsonify
from app.services.model_service import load_model, predict_from_file

predict_bp = Blueprint('predict', __name__)
model, class_names = load_model()

@predict_bp.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('image')
    if file is None:
        return jsonify({'error': 'Gambar tidak ditemukan.'}), 400

    try:
        result = predict_from_file(file, model, class_names)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

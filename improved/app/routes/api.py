from flask import Blueprint, request, jsonify
from app.services.prediction_service import PredictionService
import os

api_bp = Blueprint('api', __name__)
prediction_service = PredictionService()

def ensure_model_loaded():
    """Ensure the model is loaded, load it if not already loaded"""
    if prediction_service.model is None or prediction_service.preprocessor is None or prediction_service.graph_info is None:
        # Default paths
        default_model_path = 'models/hybrid_agricultural_model_best.pth'
        default_preprocessor_path = 'models/preprocessor.pkl'
        
        # Try to get from Flask config if available
        from flask import current_app
        try:
            model_path = current_app.config.get('MODEL_PATH', default_model_path)
            preprocessor_path = current_app.config.get('PREPROCESSOR_PATH', default_preprocessor_path)
        except RuntimeError:
            # Not in app context, use environment or defaults
            model_path = os.getenv('MODEL_PATH', default_model_path)
            preprocessor_path = os.getenv('PREPROCESSOR_PATH', default_preprocessor_path)
        
        # Verify paths exist
        if not os.path.exists(model_path):
            # Try to find model file if path is wrong
            models_dir = 'models'
            if os.path.exists(models_dir):
                model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
                if model_files:
                    model_path = os.path.join(models_dir, model_files[0])
                    print(f"⚠ Using found model file: {model_path}")
        
        if not os.path.exists(preprocessor_path):
            preprocessor_path = os.path.join('models', 'preprocessor.pkl')
        
        if os.path.exists(model_path) and os.path.exists(preprocessor_path):
            try:
                prediction_service.load_model_from_path(model_path, preprocessor_path)
                print(f"✓ Model loaded on demand (API) from {model_path}")
            except Exception as e:
                print(f"⚠ Error loading model: {e}")
                import traceback
                traceback.print_exc()
                raise
        else:
            error_msg = f"Model files not found. Model: {model_path} (exists: {os.path.exists(model_path)}), Preprocessor: {preprocessor_path} (exists: {os.path.exists(preprocessor_path)})"
            print(f"⚠ {error_msg}")
            raise ValueError(error_msg)

@api_bp.route('/predict', methods=['POST'])
def api_predict():
    try:
        # Ensure model is loaded before making predictions
        ensure_model_loaded()
        
        data = request.get_json()
        
        region = data.get('region')
        district = data.get('district')
        crop = data.get('crop')
        months_ahead = data.get('months_ahead', 1)
        
        # Use predict_individual_costs instead
        prediction = prediction_service.predict_individual_costs(
            region, district, crop
        )
        
        # Format predictions for API response
        predictions = []
        for i in range(months_ahead):
            predictions.append({
                'month': f"Month {i+1}",
                **prediction
            })
        
        return jsonify({
            'status': 'success',
            'region': region,
            'district': district,
            'crop': crop,
            'predictions': predictions,
            'single_prediction': prediction
        })
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@api_bp.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'service': 'Agricultural Cost Predictor'})

@api_bp.route('/regions', methods=['GET'])
def get_regions():
    regions = ['Central', 'Eastern', 'Northern', 'Western']
    return jsonify({'regions': regions})

@api_bp.route('/districts/<region>', methods=['GET'])
def get_districts_api(region):
    districts_map = {
        'Central': ['Kampala', 'Wakiso', 'Mukono', 'Masaka'],
        'Eastern': ['Jinja', 'Iganga', 'Mbale', 'Soroti'],
        'Northern': ['Gulu', 'Lira', 'Arua', 'Kitgum'],
        'Western': ['Mbarara', 'Fort Portal', 'Kabale', 'Hoima']
    }
    districts = districts_map.get(region, [])
    return jsonify({'districts': districts})

@api_bp.route('/crops/<district>', methods=['GET'])
def get_crops_api(district):
    crops = ['Maize', 'Beans', 'Coffee', 'Rice', 'Cassava', 'Matooke']
    return jsonify({'crops': crops})
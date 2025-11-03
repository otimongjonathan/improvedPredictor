from flask import Flask
import os

def create_app(config_name='default'):
    app = Flask(__name__)
    app.secret_key = 'agricultural_predictor_secret_key'
    
    # Load configuration
    from config import config
    config_class = config.get(config_name, config['default'])
    app.config.from_object(config_class)
    
    # Import and register blueprints
    from app.routes.views import main_bp
    from app.routes.api import api_bp
    
    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp, url_prefix='/api/v1')
    
    # Load model on startup
    try:
        from app.routes.views import prediction_service
        model_path = app.config.get('MODEL_PATH', 'models/hybrid_agricultural_model_best.pth')
        preprocessor_path = app.config.get('PREPROCESSOR_PATH', 'models/preprocessor.pkl')
        
        if os.path.exists(model_path) and os.path.exists(preprocessor_path):
            prediction_service.load_model_from_path(model_path, preprocessor_path)
            print("✓ Model loaded successfully on startup")
        else:
            print(f"⚠ Warning: Model files not found at {model_path} or {preprocessor_path}")
    except Exception as e:
        print(f"⚠ Error loading model on startup: {e}")
    
    return app
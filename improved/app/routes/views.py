from flask import Blueprint, render_template, request, jsonify
from app.services.prediction_service import PredictionService
from app.utils.farming_requirements import calculate_total_costs_per_acre, get_crop_requirements
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import os

# Define the blueprint FIRST
main_bp = Blueprint('main', __name__)
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
                print(f"✓ Model loaded on demand from {model_path}")
            except Exception as e:
                print(f"⚠ Error loading model: {e}")
                import traceback
                traceback.print_exc()
                raise
        else:
            error_msg = f"Model files not found. Model: {model_path} (exists: {os.path.exists(model_path)}), Preprocessor: {preprocessor_path} (exists: {os.path.exists(preprocessor_path)})"
            print(f"⚠ {error_msg}")
            raise ValueError(error_msg)

@main_bp.route('/')
def index():
    regions = ['Central', 'Eastern', 'Northern', 'Western']
    districts = ['Kampala', 'Wakiso', 'Jinja', 'Mbale', 'Gulu', 'Lira', 'Luweero']
    crops = ['Maize', 'Beans', 'Coffee', 'Rice', 'Cassava', 'Matooke']
    
    return render_template('index.html', 
                         regions=regions,
                         districts=districts,
                         crops=crops)

@main_bp.route('/predict', methods=['POST'])
def predict():
    try:
        # Ensure model is loaded before making predictions
        ensure_model_loaded()
        
        region = request.form['region']
        district = request.form['district']
        crop = request.form['crop']
        months_ahead = int(request.form['months_ahead'])
        
        # Use realistic default prices (no user input needed)
        current_prices = {
            'seed_price': 5000,        # UGX per kg
            'fertilizer_price': 3500,  # UGX per kg
            'herbicide_price': 12000,  # UGX per litre
            'pesticide_price': 11000,  # UGX per litre
            'labor_cost': 8000         # UGX per day
        }
        
        # Get historical data once (will be reused for all predictions)
        import pandas as pd
        train_df = pd.read_csv('train_dataset_cleaned.csv')
        historical_data = train_df[
            (train_df['Region'] == region) &
            (train_df['District'] == district) &
            (train_df['Crop'] == crop)
        ].sort_values('Date').tail(12)
        
        # Get crop requirements for calculations
        crop_requirements = get_crop_requirements(crop)
        
        # Use model to predict for each month ahead
        # Each prediction uses the model with adjusted temporal sequences
        predictions = []
        for month_num in range(1, months_ahead + 1):
            # Calculate the actual future date using proper month arithmetic
            future_date = datetime.now() + relativedelta(months=month_num)
            month_name = future_date.strftime("%B %Y")  # e.g., "November 2025"
            
            # Predict using model with temporal sequence shifted for this future month
            month_prediction = prediction_service.predict_for_future_month(
                region, district, crop, months_ahead=month_num, historical_data=historical_data.copy() if hasattr(historical_data, 'copy') else historical_data
            )
            
            # Calculate total costs per acre based on requirements
            cost_calculation = calculate_total_costs_per_acre(
                {
                    'seed_price_per_kg': month_prediction['seed_price_per_kg'],
                    'fertilizer_price_per_kg': month_prediction['fertilizer_price_per_kg'],
                    'herbicide_price_per_litre': month_prediction['herbicide_price_per_litre'],
                    'pesticide_price_per_litre': month_prediction['pesticide_price_per_litre'],
                    'labor_cost_per_day': month_prediction['labor_cost_per_day']
                },
                crop
            )
            
            # Add calculated costs to prediction
            month_prediction['month'] = month_name
            month_prediction['date'] = future_date.strftime("%Y-%m-%d")
            month_prediction['seed_quantity_per_acre'] = cost_calculation.get('seed_quantity_display', crop_requirements['seed'])
            month_prediction['seed_unit_label'] = cost_calculation.get('seed_unit_label', 'kg')
            month_prediction['fertilizer_quantity_per_acre'] = crop_requirements['fertilizer']
            month_prediction['herbicide_quantity_per_acre'] = crop_requirements['herbicide']
            month_prediction['pesticide_quantity_per_acre'] = crop_requirements['pesticide']
            month_prediction['labor_days_per_acre'] = crop_requirements['labor']
            
            # Add total costs per input type
            month_prediction['seed_total_cost'] = cost_calculation['seed_cost']
            month_prediction['fertilizer_total_cost'] = cost_calculation['fertilizer_cost']
            month_prediction['herbicide_total_cost'] = cost_calculation['herbicide_cost']
            month_prediction['pesticide_total_cost'] = cost_calculation['pesticide_cost']
            month_prediction['labor_total_cost'] = cost_calculation['labor_cost']
            month_prediction['total_cost_per_acre'] = cost_calculation['total_cost_per_acre']
            
            predictions.append(month_prediction)
        
        return render_template('results.html',
                            region=region,
                            district=district, 
                            crop=crop,
                            predictions=predictions,
                            current_prices=current_prices)
    
    except Exception as e:
        return render_template('error.html', error=str(e))

# Add other routes if needed
@main_bp.route('/get_districts/<region>')
def get_districts(region):
    districts_map = {
        'Central': ['Kampala', 'Wakiso', 'Mukono', 'Masaka'],
        'Eastern': ['Jinja', 'Iganga', 'Mbale', 'Soroti'],
        'Northern': ['Gulu', 'Lira', 'Arua', 'Kitgum'],
        'Western': ['Mbarara', 'Fort Portal', 'Kabale', 'Hoima']
    }
    districts = districts_map.get(region, [])
    return jsonify(districts)

@main_bp.route('/get_crops/<region>/<district>')
def get_crops(region, district):
    crops = ['Maize', 'Beans', 'Coffee', 'Rice', 'Cassava', 'Matooke']
    return jsonify(crops)
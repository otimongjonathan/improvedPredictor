"""
Direct test of the prediction service without Flask
"""
import sys
from app.services.prediction_service import PredictionService

def test_prediction():
    print("=" * 60)
    print("Testing Prediction Service Directly")
    print("=" * 60)
    
    # Initialize prediction service
    print("\n1. Initializing prediction service...")
    prediction_service = PredictionService()
    
    # Load model
    print("2. Loading model...")
    try:
        prediction_service.load_model_from_path(
            'models/hybrid_agricultural_model_best.pth',
            'models/preprocessor.pkl'
        )
        print("   ✓ Model loaded successfully")
    except Exception as e:
        print(f"   ✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test prediction
    print("\n3. Testing prediction...")
    try:
        # Test with Central, Kampala, Maize (common combination)
        prediction = prediction_service.predict_individual_costs(
            region='Central',
            district='Kampala',
            crop='Maize'
        )
        
        print("\n   ✓ Prediction successful!")
        print("\n   Prediction Results:")
        print(f"   Region: {prediction['region']}")
        print(f"   District: {prediction['district']}")
        print(f"   Crop: {prediction['crop']}")
        print(f"\n   Individual Costs (UGX):")
        print(f"   - Seed Price per Kg: {prediction['seed_price_per_kg']:,.2f}")
        print(f"   - Fertilizer Price per Kg: {prediction['fertilizer_price_per_kg']:,.2f}")
        print(f"   - Herbicide Price per Litre: {prediction['herbicide_price_per_litre']:,.2f}")
        print(f"   - Pesticide Price per Litre: {prediction['pesticide_price_per_litre']:,.2f}")
        print(f"   - Labor Cost per Day: {prediction['labor_cost_per_day']:,.2f}")
        print(f"\n   Total Input Cost: {prediction['total_input_cost']:,.2f} UGX")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Error making prediction: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_prediction()
    if success:
        print("\n" + "=" * 60)
        print("✓ Prediction service is working correctly!")
        print("=" * 60)
        print("\nThe app is ready to run. Start it with:")
        print("  python run.py")
        print("\nThen access it at: http://localhost:5000")
    sys.exit(0 if success else 1)


import os
from datetime import timedelta

class Config:
    """Base configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'agricultural_predictor_secret_key'
    DEBUG = False
    TESTING = False
    
    # Model configuration
    MODEL_PATH = os.environ.get('MODEL_PATH', 'models/hybrid_agricultural_model_best.pth')
    PREPROCESSOR_PATH = os.environ.get('PREPROCESSOR_PATH', 'models/preprocessor.pkl')
    DEVICE = 'cuda' if os.environ.get('USE_GPU') == 'true' else 'cpu'
    
    # Data files
    DATA_FILES = {
        'train': os.environ.get('TRAIN_DATA', 'train_dataset_cleaned.csv'),
        'test': os.environ.get('TEST_DATA', 'test_dataset_cleaned.csv'),
        'validation': os.environ.get('VALIDATION_DATA', 'validation_dataset_cleaned.csv')
    }

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    DEVELOPMENT = True

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
from app import create_app
from app.services.data_service import data_service
import os

app = create_app('production')

# Load data on startup
with app.app_context():
    try:
        data_service.load_data(app.config['DATA_FILES'])
    except Exception as e:
        print(f"⚠ Warning: Could not load data files: {e}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    app.run(host=host, port=port)
from app import create_app
import os

# Use production config for deployment, development for local
config_name = os.environ.get('FLASK_ENV', 'development')
if config_name == 'production':
    config_name = 'production'
else:
    config_name = 'development'

app = create_app(config_name)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    print(f"=== AGRICULTURAL COST PREDICTOR STARTING on {host}:{port} ===")
    app.run(host=host, port=port, debug=(config_name == 'development'))
# Railway Deployment Guide

This guide will help you deploy the Agricultural Cost Predictor to Railway.

## Prerequisites

1. A Railway account (sign up at [railway.app](https://railway.app))
2. Your code pushed to a Git repository (GitHub, GitLab, or Bitbucket)
3. Model files (`models/hybrid_agricultural_model_best.pth` and `models/preprocessor.pkl`) committed to the repository

## Deployment Steps

### 1. Connect Your Repository

1. Log in to Railway dashboard
2. Click "New Project"
3. Select "Deploy from GitHub repo" (or your Git provider)
4. Choose your repository

### 2. Configure Build Settings

Railway will automatically detect this as a Python project and use the `Procfile` for deployment. The configuration includes:

- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn wsgi:app --bind 0.0.0.0:$PORT --workers 2 --threads 2 --timeout 120`

### 3. Set Environment Variables (Optional)

You can configure these environment variables in Railway's dashboard under "Variables":

- `FLASK_ENV=production` - Sets the app to production mode
- `SECRET_KEY=<your-secret-key>` - Flask secret key (generate a secure random string)
- `MODEL_PATH=models/hybrid_agricultural_model_best.pth` - Path to model file (default)
- `PREPROCESSOR_PATH=models/preprocessor.pkl` - Path to preprocessor (default)
- `USE_GPU=false` - Set to `true` if Railway provides GPU support (currently unlikely)

### 4. Verify Model Files Are Included

Ensure these files are in your repository:
- `models/hybrid_agricultural_model_best.pth`
- `models/preprocessor.pkl`

Railway will automatically exclude files listed in `.railwayignore`.

### 5. Deploy

Railway will automatically:
1. Install dependencies from `requirements.txt`
2. Start the application using the `Procfile`
3. Make it available at a Railway-provided URL

### 6. Monitor Deployment

- Check the deployment logs in Railway dashboard
- Look for "✓ Model loaded successfully on startup" message
- Test the application at your Railway-provided URL

## File Structure for Deployment

```
agricultural-predictor/
├── Procfile                 # Railway start command
├── railway.json            # Railway configuration
├── wsgi.py                 # WSGI entry point
├── run.py                  # Local development entry point
├── requirements.txt        # Python dependencies
├── config.py              # Application configuration
├── .railwayignore         # Files to exclude from deployment
├── app/                   # Application code
│   ├── __init__.py
│   ├── routes/
│   ├── services/
│   ├── models/
│   ├── utils/
│   └── templates/
└── models/                # Model files (MUST be in repo)
    ├── hybrid_agricultural_model_best.pth
    └── preprocessor.pkl
```

## Important Notes

1. **Model Files**: Ensure model files are committed to your repository or use Railway's persistent storage
2. **Memory**: This application uses PyTorch which can be memory-intensive. Monitor Railway logs for memory issues
3. **Startup Time**: First deployment may take longer as dependencies (especially PyTorch) are installed
4. **Port Binding**: The app automatically binds to the `PORT` environment variable provided by Railway

## Troubleshooting

### Model Not Loading
- Check that model files exist in `models/` directory
- Verify file paths in Railway logs
- Ensure files are not in `.railwayignore`

### Memory Issues
- Railway free tier has limited memory
- Consider upgrading if you experience OOM errors
- Reduce gunicorn workers in `Procfile` if needed

### Build Fails
- Check Railway build logs
- Verify all dependencies in `requirements.txt`
- Ensure Python version compatibility (Railway uses Python 3.9+ by default)

### App Crashes on Startup
- Check application logs in Railway dashboard
- Verify environment variables are set correctly
- Ensure all required files are present

## Custom Domain

After deployment, you can add a custom domain:
1. Go to your project settings in Railway
2. Click "Settings" → "Domains"
3. Add your custom domain
4. Railway will provide DNS configuration

## Updates

To update the application:
1. Push changes to your Git repository
2. Railway will automatically detect changes and redeploy
3. Monitor the deployment in Railway dashboard


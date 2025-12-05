# Neural BUS Web Demo

Interactive web demo for the Neural BUS project - Visual Question Answering powered by BLIP.

## Project Structure

```
neural-bus-deploy/
├── app.py                 # Flask backend
├── requirements.txt       # Python dependencies
├── render.yaml           # Render deployment config
├── templates/
│   └── demo.html         # Frontend interface
└── assets/               # Static assets (if needed)
```

## Local Development

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run the app:**
```bash
python app.py
```

3. **Open browser:**
Navigate to `http://localhost:5001`

## Deployment on Render

### Option 1: Using render.yaml (Recommended)

1. **Push to GitHub:**
```bash
git init
git add .
git commit -m "Initial commit: Neural BUS demo"
git remote add origin YOUR_GITHUB_REPO_URL
git push -u origin main
```

2. **Connect to Render:**
   - Go to https://render.com
   - Click "New +" → "Blueprint"
   - Connect your GitHub repository
   - Render will auto-detect `render.yaml` and deploy

### Option 2: Manual Setup

1. **Push code to GitHub** (same as above)

2. **Create Web Service on Render:**
   - Go to https://render.com
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Configure:
     - **Name:** neural-bus-demo
     - **Environment:** Python 3
     - **Build Command:** `pip install -r requirements.txt`
     - **Start Command:** `gunicorn app:app`
     - **Instance Type:** Free (or higher for better performance)

3. **Deploy:**
   - Click "Create Web Service"
   - Wait for build and deployment (first time takes ~5-10 minutes due to model download)

## Important Notes

⚠️ **First Deployment:** The initial deployment takes longer because the BLIP model needs to be downloaded (~500MB). This happens once during the build.

⚠️ **Free Tier Limitations:** Render's free tier:
- Spins down after 15 minutes of inactivity
- First request after spin-down will be slow (~30-60 seconds)
- Consider upgrading for production use

⚠️ **Memory Requirements:** The BLIP model requires at least 1GB RAM. Free tier should work, but consider upgrading if you face issues.

## API Endpoints

- `GET /` - Frontend interface
- `POST /api/process` - Process VQA request
  ```json
  {
    "image": "data:image/png;base64,...",
    "question": "What is in the image?"
  }
  ```
- `GET /health` - Health check

## Deployment on Vercel (Alternative)

Vercel is more complex for Flask apps. If you prefer Vercel:

1. **Create `vercel.json`:**
```json
{
  "version": 2,
  "builds": [
    {
      "src": "app.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "app.py"
    }
  ]
}
```

2. **Modify app.py:** Add at the end:
```python
# For Vercel
app = app
```

3. **Deploy:**
```bash
vercel --prod
```

⚠️ **Note:** Vercel has serverless function size limits that may not work well with large ML models. Render is recommended for this project.

## Troubleshooting

### Model loading fails
- Check logs: Transformers library might need more time to download
- Increase timeout settings in Render dashboard

### Out of memory
- Upgrade to a paid plan with more RAM
- Consider using CPU-optimized instance

### Slow response times
- Expected on first request after deployment/spin-down
- Consider keeping a paid instance always running

## Features

- 📸 Drag & drop image upload
- 💬 Visual Question Answering
- 🚌 Neural BUS protocol visualization
- ⚡ Real-time pipeline tracking
- 📊 Performance metrics display

## Technology Stack

- **Backend:** Flask + Gunicorn
- **ML Model:** BLIP VQA (Salesforce/blip-vqa-base)
- **Frontend:** Vanilla HTML/CSS/JavaScript
- **Deployment:** Render / Vercel

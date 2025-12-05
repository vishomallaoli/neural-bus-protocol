# Quick Deployment Guide

## ✅ Recommended: Render (Best for ML Models)

### Step-by-Step Instructions:

1. **Prepare Your Repository:**
   ```bash
   cd neural-bus-deploy
   git init
   git add .
   git commit -m "Initial commit"
   ```

2. **Push to GitHub:**
   - Create a new repository on GitHub
   - Link and push:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/neural-bus-demo.git
   git branch -M main
   git push -u origin main
   ```

3. **Deploy on Render:**
   - Visit https://render.com and sign up/login
   - Click "New +" → "Blueprint" (if using render.yaml)
   - OR click "New +" → "Web Service" (manual setup)
   - Connect your GitHub account
   - Select your repository
   - Render will auto-detect Python and deploy!

4. **Configuration (if manual setup):**
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
   - **Environment:** Python 3
   - **Instance Type:** Free (or paid for better performance)

5. **Wait for Deployment:**
   - First deployment: ~5-10 minutes (model download)
   - Subsequent deployments: ~2-3 minutes
   - You'll get a URL like: `https://neural-bus-demo.onrender.com`

### Important Render Notes:
- ⏱️ Free tier spins down after 15 min inactivity
- 🐌 First request after spin-down is slow (~30-60s)
- 💾 Requires ~1GB RAM (free tier should work)
- 💰 Consider paid plan ($7/mo) for always-on service

---

## Alternative: Vercel (Not Recommended for Large Models)

⚠️ **Warning:** Vercel has strict function size limits (50MB) which may not accommodate the BLIP model (~500MB). Use only if you can optimize the model size.

1. **Install Vercel CLI:**
   ```bash
   npm install -g vercel
   ```

2. **Deploy:**
   ```bash
   cd neural-bus-deploy
   vercel --prod
   ```

3. **Configure:**
   - Follow prompts
   - Vercel will use `vercel.json` automatically

---

## Alternative: Heroku

1. **Install Heroku CLI:**
   ```bash
   # macOS
   brew tap heroku/brew && brew install heroku
   
   # Others: https://devcenter.heroku.com/articles/heroku-cli
   ```

2. **Login and Create App:**
   ```bash
   heroku login
   cd neural-bus-deploy
   heroku create neural-bus-demo
   ```

3. **Deploy:**
   ```bash
   git push heroku main
   ```

4. **Open:**
   ```bash
   heroku open
   ```

---

## Testing Locally First

**Always test locally before deploying:**

```bash
cd neural-bus-deploy

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run
python app.py

# Test in browser
open http://localhost:5001
```

---

## Troubleshooting

### "Model loading failed"
- Check logs for download errors
- Verify internet connection during deployment
- Increase build timeout in platform settings

### "Out of memory"
- Upgrade to paid tier with more RAM
- Render: Use at least 1GB instance
- Optimize model loading (use quantized models)

### "502 Bad Gateway / Timeout"
- First request takes time (model loading)
- Wait 30-60 seconds and retry
- Consider health check warmup

### "Port binding error"
- Render/Heroku auto-set PORT env variable
- Flask should use: `port=int(os.environ.get("PORT", 5001))`

---

## Monitoring

After deployment, monitor:
- Response times
- Error rates
- Memory usage
- Cold start times

Most platforms provide built-in dashboards for these metrics.

---

## Next Steps

Once deployed:
1. ✅ Test with sample images
2. ✅ Share the URL with your advisor/team
3. ✅ Add to your project documentation
4. ✅ Include in your capstone presentation

Good luck with your deployment! 🚀

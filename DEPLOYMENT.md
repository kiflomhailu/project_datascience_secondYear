# Seismic Risk Monitoring Dashboard - Render Deployment

## 🚀 Quick Deploy to Render

### Prerequisites
- GitHub account
- Render account (free tier works!)
- Push this repository to GitHub

### Deployment Steps

#### 1. Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit - Seismic dashboard"
git branch -M main
git remote add origin YOUR_GITHUB_REPO_URL
git push -u origin main
```

#### 2. Deploy on Render

**Option A: Blueprint Deploy (Recommended)**
1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click **"New"** → **"Blueprint"**
3. Connect your GitHub repository
4. Render will auto-detect `render.yaml` and create both services:
   - `seismic-dashboard-api` (Flask backend)
   - `seismic-dashboard-frontend` (Static site)

**Option B: Manual Deploy**

**Backend API:**
1. New → Web Service
2. Connect repository
3. Settings:
   - **Name**: `seismic-dashboard-api`
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r dashboard/api/requirements.txt`
   - **Start Command**: `gunicorn -w 4 -b 0.0.0.0:$PORT dashboard.api.app:app`
   - **Instance Type**: Free

**Frontend:**
1. New → Static Site
2. Connect repository
3. Settings:
   - **Name**: `seismic-dashboard-frontend`
   - **Publish Directory**: `dashboard`
   - **Instance Type**: Free

#### 3. Update Frontend API URL

After backend deploys, copy the API URL (e.g., `https://seismic-dashboard-api.onrender.com`)

Update `dashboard/index.html` line ~1940:
```javascript
const API_BASE_URL = 'https://YOUR-API-URL.onrender.com';
```

Commit and push the change:
```bash
git add dashboard/index.html
git commit -m "Update API URL for production"
git push
```

#### 4. Access Your Dashboard

Frontend URL: `https://seismic-dashboard-frontend.onrender.com`

---

## 📝 Important Notes

### Free Tier Limitations
- Services spin down after 15 minutes of inactivity
- First request after spin-down takes ~30-60 seconds
- 750 hours/month free (enough for 1 service 24/7)

### Model Files
Ensure these files are in the `latest/` folder:
- `seismic_event_occurrence_model_v2.cbm`
- `seismic_magnitude_model_v2.cbm`
- `seismic_traffic_light_3class_model_v2.cbm`
- `train_medians_v2.pkl`
- `optimal_event_threshold_v2.txt`
- `operational_seismic_linear_decay121.csv`

### Environment Variables (Optional)
On Render dashboard, you can set:
- `FLASK_ENV=production`
- `PYTHON_VERSION=3.11.0`

---

## 🔧 Local Testing Before Deploy

Test the production setup locally:

```bash
# Install dependencies
pip install -r dashboard/api/requirements.txt

# Run with gunicorn (production server)
gunicorn -w 4 -b 0.0.0.0:5000 dashboard.api.app:app

# In another terminal, serve frontend
cd dashboard
python -m http.server 8080
```

Visit: http://localhost:8080

---

## 🐛 Troubleshooting

### Build Fails
- Check all model files exist in `latest/` folder
- Verify `requirements.txt` is correct
- Check Render build logs

### API Returns 404
- Verify API service is running
- Check API URL in frontend matches backend URL
- Check CORS settings in `app.py`

### Slow Performance
- Free tier spins down after inactivity
- Consider upgrading to paid tier ($7/month) for persistent service
- First load after spin-down takes 30-60 seconds

### Models Not Loading
- Ensure all `.cbm` files are committed to git
- Check file paths in `app.py`
- Verify build logs show successful model loading

---

## 💰 Cost Estimate

**Free Tier:**
- ✅ 1 Web Service (API): Free
- ✅ 1 Static Site: Free
- ⚠️ Spins down after 15 min inactivity

**Paid Tier ($7/month per service):**
- ✅ Always-on, no spin-down
- ✅ Better performance
- ✅ Custom domains

---

## 🔒 Security Recommendations

After deployment, update CORS in `dashboard/api/app.py`:

```python
CORS(app, resources={
    r"/*": {
        "origins": ["https://YOUR-FRONTEND-URL.onrender.com"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})
```

---

## 📚 Additional Resources

- [Render Documentation](https://render.com/docs)
- [Flask Deployment Guide](https://flask.palletsprojects.com/en/latest/deploying/)
- [Gunicorn Configuration](https://docs.gunicorn.org/en/stable/configure.html)

---

## ✅ Deployment Checklist

- [ ] Repository pushed to GitHub
- [ ] All model files in `latest/` folder
- [ ] `render.yaml` configured
- [ ] Backend deployed on Render
- [ ] Frontend deployed on Render
- [ ] API URL updated in `index.html`
- [ ] Dashboard accessible and working
- [ ] CORS configured for production domain

---

**Happy Deploying! 🎉**

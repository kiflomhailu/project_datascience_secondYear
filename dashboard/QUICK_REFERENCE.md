# ⚡ Quick Reference Guide

## 🎯 Common Tasks

### Start Everything Locally
```bash
# Terminal 1: Start API
cd api && python app.py

# Terminal 2: Start Dashboard
cd dashboard && python -m http.server 8080

# Browser: Open http://localhost:8080
```

### Test API
```bash
# Health check
curl http://localhost:5000/health

# Should return: {"status":"healthy","model_loaded":true}
```

### Update API URL
Edit `index.html` line 250:
- Local: `http://localhost:5000`
- Cloud: `https://project-datascience-secondyear.onrender.com`

---

## 🔗 Important Links

### Local Development
- Dashboard: `http://localhost:8080`
- API: `http://localhost:5000`
- API Health: `http://localhost:5000/health`

### Cloud Deployment
- Dashboard: `https://kiflomhailu.github.io/project_datascience_secondYear/dashboard/`
- API: `https://project-datascience-secondyear.onrender.com`
- API Health: `https://project-datascience-secondyear.onrender.com/health`

---

## 🐛 Common Issues

### "API Not Connected"
- ✅ Check API is running: `python app.py` in `api/` folder
- ✅ Check port 5000 is not blocked
- ✅ Check browser console (F12) for errors

### "No Data Available"
- ✅ **Local:** Check CSV files in `dashboard/data/` folder
- ✅ **Cloud:** Expected (data files not included for security)
- ✅ Model predictions still work without data files

### "Model Not Found"
- ✅ Check `lstm_model_ammad.h5` is in `dashboard/` folder
- ✅ Check API logs for file path errors

### Charts Show 000
- ✅ Check API is returning data (not empty arrays)
- ✅ Check date range matches data range
- ✅ Check browser console for errors

---

## 📝 File Locations

| File | Location |
|------|----------|
| Model | `dashboard/lstm_model_ammad.h5` |
| API Code | `dashboard/api/app.py` |
| Dashboard | `dashboard/index.html` |
| Data (local) | `dashboard/data/*.csv` |
| Config | `dashboard/index.html` (line 250) |

---

## 🔄 Git Commands

```bash
# Check status
git status

# Add changes
git add .

# Commit
git commit -m "Your message"

# Push
git push origin main
```

---

## 📊 Dashboard Features

### Operational Dashboard
- Real-time metrics visualization
- Date/time range selection
- Metric checkboxes (14 options)
- Chart controls (zoom, pan, screenshot)

### Risk Dashboard
- 7-day risk forecast
- LSTM model predictions
- Risk level probabilities
- Color-coded alerts

---

## 🚀 Deployment Checklist

- [ ] API deployed on Render
- [ ] Model file in repository
- [ ] Dashboard deployed on GitHub Pages
- [ ] API URL updated in `index.html`
- [ ] Test health endpoint
- [ ] Test dashboard connection

---

## 💡 Tips

1. **For Presentation:** Use local setup with data files
2. **For Demo:** Use cloud deployment (works without data)
3. **For Development:** Use local for faster iteration
4. **Always test:** Check browser console (F12) for errors


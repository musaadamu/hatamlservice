# 🚀 Deploying HATA ML Service to Render Free Tier

## 📋 Overview

This ML service is optimized for Render's **free tier (512MB RAM)** by using HuggingFace's Inference API instead of loading the model locally.

---

## ✅ Prerequisites

1. **HuggingFace Account** - [Sign up here](https://huggingface.co/join)
2. **HuggingFace Token** - [Create token here](https://huggingface.co/settings/tokens)
   - Token type: **Read** access is sufficient
3. **Render Account** - [Sign up here](https://render.com)

---

## 🔧 Deployment Steps

### **Step 1: Get Your HuggingFace Token**

1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Name: `HATA-ML-Service`
4. Type: **Read**
5. Click "Generate"
6. **Copy the token** (you'll need it in Step 3)

### **Step 2: Push Code to GitHub**

```bash
git add .
git commit -m "Optimize ML service for Render free tier"
git push origin main
```

### **Step 3: Deploy to Render**

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Select the `hata` repository
5. Configure:
   - **Name**: `hatamlservice`
   - **Root Directory**: `ml-service`
   - **Runtime**: Python 3
   - **Build Command**: `pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port 5000`
   - **Plan**: Free

6. **Add Environment Variable**:
   - Click "Advanced" → "Add Environment Variable"
   - Key: `HF_TOKEN`
   - Value: *[Paste your HuggingFace token from Step 1]*

7. Click "Create Web Service"

### **Step 4: Wait for Deployment**

- Initial deployment takes ~5-10 minutes
- Watch the logs for:
  ```
  ✅ Build successful
  🌐 Using HuggingFace Inference API mode
  ```

### **Step 5: Test the Service**

Once deployed, test the health endpoint:

```bash
curl https://your-service-name.onrender.com/health
```

Expected response:
```json
{
  "status": "healthy",
  "inference_type": "huggingface_api",
  "model_name": "msmaje/Quantizedphdhatamodel"
}
```

---

## 📊 Memory Usage

| Component | RAM Usage |
|-----------|-----------|
| **FastAPI + Uvicorn** | ~50MB |
| **LIME + NumPy** | ~100MB |
| **HTTP Clients** | ~20MB |
| **Python Runtime** | ~80MB |
| **Total** | **~250MB** ✅ |

**Remaining headroom**: ~260MB for request processing

---

## ⚡ Performance Expectations

### **With HuggingFace Inference API:**

| Text Length | Prediction Time | LIME Explanation Time | Total Time |
|-------------|-----------------|----------------------|------------|
| Short (100 chars) | ~1-2 seconds | ~30 seconds | **~32 seconds** |
| Medium (500 chars) | ~2-3 seconds | ~2 minutes | **~2.5 minutes** |
| Long (1000 chars) | ~3-5 seconds | ~3 minutes | **~3.5 minutes** |

**Note**: First request after deployment may take 20-30 seconds as HuggingFace loads the model.

---

## 🐛 Troubleshooting

### **Issue: Out of Memory**

**Symptoms**: Service crashes with "Out of memory (used over 512Mi)"

**Solutions**:
1. Verify `USE_HF_INFERENCE_API=True` in config
2. Check `requirements.txt` doesn't include PyTorch/Transformers
3. Reduce `LIME_NUM_SAMPLES` to 50 in config

### **Issue: HuggingFace API Errors**

**Symptoms**: 401 Unauthorized or 403 Forbidden

**Solutions**:
1. Verify `HF_TOKEN` is set in Render environment variables
2. Check token has **Read** access
3. Ensure model `msmaje/Quantizedphdhatamodel` is public

### **Issue: Slow First Request**

**Symptoms**: First prediction takes 20-30 seconds

**Explanation**: HuggingFace Inference API "cold starts" the model on first request. Subsequent requests are fast.

**Solution**: This is normal behavior. Consider implementing a warmup request on startup.

---

## 🔄 Updating the Service

```bash
# Make changes to code
git add .
git commit -m "Update ML service"
git push origin main

# Render will auto-deploy if auto-deploy is enabled
# Or manually deploy from Render dashboard
```

---

## 💰 Cost Optimization

### **Free Tier Limits:**
- ✅ 512MB RAM
- ✅ Shared CPU
- ✅ 750 hours/month
- ✅ Sleeps after 15 min inactivity

### **Staying Within Limits:**
1. ✅ Use HuggingFace API (no model in RAM)
2. ✅ Minimal dependencies
3. ✅ Efficient LIME sampling (100 samples)
4. ✅ No caching (saves RAM)

---

## 📞 Support

If you encounter issues:
1. Check Render logs: Dashboard → Your Service → Logs
2. Check ML service logs in the logs tab
3. Verify environment variables are set correctly

---

## 🎉 Success Checklist

- [ ] HuggingFace token created
- [ ] Token added to Render environment variables
- [ ] Service deployed successfully
- [ ] Health check returns "healthy"
- [ ] Test prediction completes successfully
- [ ] Memory usage stays under 512MB

---

**Your ML service is now running on Render's free tier!** 🚀


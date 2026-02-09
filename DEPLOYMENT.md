# 🚀 Deployment Guide: Streamlit Cloud (Free Hosting)

This guide will help you deploy your Resume Optimizer app on Streamlit Cloud completely free.

## 📋 Prerequisites

1. **GitHub Account**: Create one at [github.com](https://github.com)
2. **Streamlit Account**: Sign up at [share.streamlit.io](https://share.streamlit.io/)
3. **Working App**: Ensure your app runs locally

## 🛠️ Step 1: Prepare for Cloud Deployment

### Update App for Cloud Compatibility

Since Streamlit Cloud doesn't have Ollama, you need to modify the app:

#### Option A: Use OpenAI API (Recommended)
```python
# Add to requirements.txt
openai==1.3.0

# Modify streamlit_app.py
import openai

# Replace Ollama function
def query_ollama(prompt, max_retries=3):
    try:
        client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert resume writer and career coach."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024,
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None
```

#### Option B: Use Hugging Face API (Free Tier Available)
```python
# Add to requirements.txt
transformers==4.35.0
torch==2.1.0

# Modify streamlit_app.py
from transformers import pipeline

def query_ollama(prompt, max_retries=3):
    try:
        generator = pipeline(
            "text-generation",
            model="microsoft/DialoGPT-medium",
            token=st.secrets["HF_TOKEN"]
        )
        
        response = generator(prompt, max_length=1024, temperature=0.2)
        return response[0]["generated_text"]
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None
```

### Create Secrets File

Create `.streamlit/secrets.toml` locally:

```toml
# For OpenAI
OPENAI_API_KEY = "your-openai-key-here"

# For Hugging Face
HF_TOKEN = "your-huggingface-token-here"
```

## 📤 Step 2: Push to GitHub

### Initialize Git Repository
```bash
git init
git add .
git commit -m "Initial commit: Resume Optimizer Streamlit App"
```

### Create GitHub Repository
1. Go to [GitHub](https://github.com) and create new repository
2. Name it `resume-optimizer`
3. Don't initialize with README (we already have one)

### Push to GitHub
```bash
git remote add origin https://github.com/yourusername/resume-optimizer.git
git branch -M main
git push -u origin main
```

## ☁️ Step 3: Deploy to Streamlit Cloud

### Deploy Your App
1. **Go to Streamlit Cloud**: [share.streamlit.io](https://share.streamlit.io/)
2. **Sign In**: Use your GitHub account
3. **Click "New app"** button
4. **Configure Deployment**:
   - **Repository**: Select `resume-optimizer`
   - **Branch**: Select `main`
   - **Main file**: Select `streamlit_app.py`
   - **App URL**: Will be auto-generated
5. **Click "Deploy"**

### Add Secrets to Streamlit Cloud
1. **Go to your app dashboard** on Streamlit Cloud
2. **Click "⋮" menu** → "Settings"
3. **Go to "Secrets" tab**
4. **Add your API keys**:
   ```
   # For OpenAI
   OPENAI_API_KEY=sk-your-actual-openai-key
   
   # For Hugging Face
   HF_TOKEN=hf_your-actual-huggingface-token
   ```
5. **Save secrets**

## 🎯 Step 4: Verify Deployment

### Check Your App
- **URL**: `https://yourusername-resume-optimizer.streamlit.app`
- **Test**: Upload a resume and run analysis
- **Monitor**: Check for any errors in Streamlit Cloud logs

### Common Cloud Issues & Solutions

#### 1. Model Loading Error
```
Error: "CUDA out of memory" or "Model too large"
```
**Solution**: Use smaller models
```python
# Change model in app
MODEL_NAME = "llama3.2:3b"  # Instead of 70b
```

#### 2. API Key Issues
```
Error: "Invalid API key" or "Rate limit exceeded"
```
**Solution**: 
- Verify API key in secrets
- Check API usage limits
- Use paid tier if needed

#### 3. Slow Performance
```
App takes too long to load
```
**Solution**:
- Use `@st.cache_resource` for models
- Reduce input text size
- Optimize image sizes

## 🆓 Free Tier Limitations

### Streamlit Cloud Free Tier
- ✅ **Public apps**: Unlimited
- ✅ **Community support**: Available
- ✅ **Basic compute**: Shared resources
- ❌ **Private apps**: Not available
- ❌ **Custom domains**: Not available
- ❌ **Dedicated resources**: Not available

### API Costs
- **OpenAI**: $0.002 per 1K tokens (free $5 credit)
- **Hugging Face**: Free tier with rate limits
- **Local Ollama**: Free but requires self-hosting

## 🔄 Step 5: Update and Maintain

### Updating Your App
```bash
# Make changes locally
git add .
git commit -m "Update: Add new feature"
git push origin main

# Streamlit Cloud auto-redeploys on push
```

### Monitoring
- **Logs**: Check Streamlit Cloud dashboard
- **Analytics**: Streamlit provides basic usage stats
- **User Feedback**: Add contact form in app

## 📱 Alternative Free Hosting Options

### 1. Hugging Face Spaces
- **Free tier**: Public spaces
- **Hardware**: CPU basic (upgradeable)
- **URL**: `https://huggingface.co/spaces/yourusername/resume-optimizer`

### 2. Replit
- **Free tier**: 750 hours/month
- **Always on**: Not available in free tier
- **URL**: `https://resume-optimizer.yourusername.repl.co`

### 3. Railway
- **Free tier**: $5 credit monthly
- **Auto-deploy**: From GitHub
- **URL**: Custom subdomain

## 🎉 Success Checklist

- [ ] App runs locally without errors
- [ ] All dependencies in requirements.txt
- [ ] API keys configured in secrets
- [ ] Repository pushed to GitHub
- [ ] App deployed on Streamlit Cloud
- [ ] Test upload and analysis works
- [ ] Download functionality works
- [ ] Mobile responsive design

## 🆘 Troubleshooting

### App Won't Deploy
1. Check requirements.txt format
2. Verify main file path
3. Check for syntax errors
4. Review Streamlit Cloud logs

### Runtime Errors
1. Check API key configuration
2. Verify model availability
3. Monitor resource usage
4. Test with smaller inputs

### Performance Issues
1. Optimize model loading
2. Reduce computational complexity
3. Implement caching
4. Use smaller models

---

**🎊 Congratulations! Your Resume Optimizer is now live and free for everyone to use!**

## 📞 Support

- **Streamlit Docs**: https://docs.streamlit.io/
- **GitHub Issues**: Create issue in your repository
- **Community**: Streamlit Discord community

# Resume Optimizer & Interview Question Generator

A powerful Streamlit application that optimizes resumes based on job descriptions and generates role-specific interview questions using AI.

## 🚀 Features

- **Resume Upload**: Support for PDF, DOCX, and TXT files
- **AI-Powered Optimization**: Enhances resumes to match job requirements
- **Interview Questions**: Generates behavioral, technical, and situational questions
- **Match Scoring**: Calculates similarity between resume and job description
- **Vector Search**: Uses FAISS for semantic content retrieval
- **Interactive Interface**: Edit and customize optimized content
- **Download Results**: Export optimized resume and interview questions

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **AI/LLM**: LLaMA 3.2 (via Ollama)
- **Embeddings**: Sentence Transformers (BAAI/bge-small-en-v1.5)
- **Vector Database**: FAISS
- **Document Processing**: PyPDF2, python-docx

## 📋 Prerequisites

1. **Python 3.8+**
2. **Ollama** installed and running:
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Start Ollama server
   ollama serve
   
   # Pull the model
   ollama pull llama3.2:3b
   ```

## 🚀 Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/resume-optimizer.git
cd resume-optimizer
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`

## 🌐 Free Hosting on Streamlit Cloud

### Step 1: Prepare Your Repository

1. **Create GitHub Repository**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/yourusername/resume-optimizer.git
   git push -u origin main
   ```

2. **Create requirements.txt** (already included):
   ```txt
   streamlit==1.28.1
   numpy==1.24.3
   requests==2.31.0
   sentence-transformers==2.2.2
   faiss-cpu==1.7.4
   python-docx==0.8.11
   PyPDF2==3.0.1
   ```

### Step 2: Deploy to Streamlit Cloud

1. **Sign Up/Sign In**:
   - Go to [Streamlit Cloud](https://share.streamlit.io/)
   - Sign up with GitHub account

2. **Deploy Your App**:
   - Click "New app" button
   - Select your GitHub repository
   - Choose the branch (usually `main`)
   - Select `streamlit_app.py` as main file
   - Click "Deploy"

3. **Configuration**:
   - **App URL**: Your app will be available at `https://yourusername-resume-optimizer.streamlit.app`
   - **Free Tier**: Includes public apps, community support
   - **Resources**: Shared compute resources

### Step 3: Handle Ollama Dependency

Since Streamlit Cloud doesn't have Ollama pre-installed, you have two options:

#### Option A: Use External API (Recommended)
Modify the app to use a hosted LLM API:

```python
# Replace Ollama calls with API calls
import openai  # or huggingface_hub, etc.

def query_llm(prompt):
    # Use OpenAI, Hugging Face, or other API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

#### Option B: Use Streamlit's Built-in LLM
```python
# Use Streamlit's experimental LLM support
import streamlit as st

def query_llm(prompt):
    # This requires Streamlit 1.28.1+
    return st.experimental_llm.chat_completion(prompt)
```

## 📁 Project Structure

```
resume-optimizer/
├── streamlit_app.py          # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md               # This file
└── .gitignore             # Git ignore file
```

## 🎯 How to Use

1. **Upload Resume**: 
   - Click "Choose your resume file"
   - Supports PDF, DOCX, TXT formats
   - Maximum file size: 10MB

2. **Enter Job Description**:
   - Paste complete job description
   - Include requirements and responsibilities
   - More details = better optimization

3. **Run Analysis**:
   - Click "Run Full Analysis"
   - Wait for AI processing (30-60 seconds)

4. **Review Results**:
   - **Match Score**: See how well your resume matches the job
   - **Optimized Resume**: AI-enhanced version
   - **Interview Questions**: Role-specific preparation questions

5. **Customize & Download**:
   - Edit optimized resume as needed
   - Download final versions
   - Prepare for your interview!

## 🔧 Configuration

### Model Settings
Edit these variables in `streamlit_app.py`:

```python
# Change model as needed
MODEL_NAME = "llama3.2:3b"  # or "llama2", "mistral", etc.

# Adjust Ollama URL if running on different server
OLLAMA_URL = "http://localhost:11434/api/generate"
```

### Customization Options
- **Embedding Model**: Change `BAAI/bge-small-en-v1.5` to other models
- **Chunk Size**: Adjust text chunking for better results
- **Temperature**: Modify AI creativity (0.0-1.0)

## 🐛 Troubleshooting

### Common Issues

1. **Ollama Connection Error**:
   ```bash
   # Check if Ollama is running
   curl http://localhost:11434/api/tags
   
   # Restart Ollama if needed
   ollama serve
   ```

2. **Model Not Found**:
   ```bash
   # List available models
   ollama list
   
   # Pull required model
   ollama pull llama3.2:3b
   ```

3. **Memory Issues**:
   - Use smaller models (`llama3.2:3b` instead of `llama3.2:70b`)
   - Close other applications
   - Reduce chunk size in code

4. **Slow Performance**:
   - Use CPU-optimized models
   - Reduce text input size
   - Check internet connection

### Performance Tips

- **For Production**: Use GPU-enabled Ollama
- **For Speed**: Cache embeddings model (already implemented)
- **For Quality**: Use larger models with more parameters

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes and test
4. Commit: `git commit -m "Add feature"`
5. Push: `git push origin feature-name`
6. Submit Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) - Web app framework
- [Ollama](https://ollama.ai/) - Local LLM serving
- [Sentence Transformers](https://www.sbert.net/) - Text embeddings
- [FAISS](https://faiss.ai/) - Vector similarity search

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/resume-optimizer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/resume-optimizer/discussions)
- **Email**: your.email@example.com

---

**⭐ If this project helps you, please give it a star on GitHub!**

## 🔗 Quick Links

- **Live Demo**: https://yourusername-resume-optimizer.streamlit.app
- **GitHub Repository**: https://github.com/yourusername/resume-optimizer
- **Streamlit Documentation**: https://docs.streamlit.io/
- **Ollama Documentation**: https://ollama.ai/documentation

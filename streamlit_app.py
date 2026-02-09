import streamlit as st
import numpy as np
import io
import requests
import docx
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import tempfile
import os

# Page configuration
st.set_page_config(
    page_title="Resume Optimizer & Interview Generator",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"

# Load embeddings model
@st.cache_resource
def load_embed_model():
    return SentenceTransformer("BAAI/bge-small-en-v1.5")

embed_model = load_embed_model()

# Session state initialization
if "resume_text" not in st.session_state:
    st.session_state.resume_text = None
if "vector_store_chunks" not in st.session_state:
    st.session_state.vector_store_chunks = None
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

# Utility functions
def read_docx(file_bytes):
    """Read DOCX file content"""
    doc = docx.Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs)

def read_pdf(file_bytes):
    """Read PDF file content"""
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_file(uploaded_file) -> str:
    """Extract text from uploaded file (PDF or DOCX)"""
    file_bytes = uploaded_file.read()
    
    if uploaded_file.name.endswith('.docx'):
        return read_docx(file_bytes)
    elif uploaded_file.name.endswith('.pdf'):
        return read_pdf(file_bytes)
    elif uploaded_file.name.endswith('.txt'):
        return file_bytes.decode('utf-8')
    else:
        st.error("Unsupported file format. Please upload PDF, DOCX, or TXT files.")
        return None

def check_ollama_connection():
    """Check if Ollama is running and responsive"""
    try:
        response = requests.get(f"{OLLAMA_URL.replace('/api/generate', '/api/tags')}", timeout=5)
        return response.status_code == 200
    except:
        return False

def query_ollama(prompt, max_retries=3):
    """Query Ollama with robust timeout and retry handling"""
    if not check_ollama_connection():
        st.error("Ollama service is not available. Please ensure Ollama is running.")
        return None
    
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 1024
        }
    }
    
    for attempt in range(max_retries):
        try:
            timeout = 60 + (attempt * 30)
            response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json()["response"]
            
        except requests.exceptions.ReadTimeout:
            if attempt == max_retries - 1:
                st.error(f"Request timed out after {max_retries} attempts.")
            
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to Ollama. Ensure Ollama is running with 'ollama serve'")
            return None
            
        except requests.exceptions.HTTPError as e:
            st.error(f"Ollama server error: {e}")
            return None
            
        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"Unexpected error: {str(e)}")
                return None
    
    return None

def chunk_text(text, chunk_size=400):
    """Improved chunking that preserves page boundaries"""
    if not text or not text.strip():
        return []
    
    # Split by page markers first
    pages = text.split("--- Page")
    
    all_chunks = []
    
    for i, page in enumerate(pages):
        if not page.strip():
            continue
            
        # Add the page marker back
        page_text = f"--- Page{page}" if i > 0 else page
        
        # Split page into chunks
        words = page_text.split()
        page_chunks = [" ".join(words[j:j+chunk_size]) for j in range(0, len(words), chunk_size)]
        
        all_chunks.extend(page_chunks)
    
    return all_chunks

def build_vector_store(text):
    """Build vector store from text chunks"""
    chunks = chunk_text(text)
    if not chunks:
        return False
    
    embeddings = embed_model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    
    st.session_state.vector_store_chunks = chunks
    st.session_state.faiss_index = index
    return True

def retrieve_chunks(query, k=4):
    """Retrieve relevant chunks based on query"""
    if st.session_state.faiss_index is None:
        return []
    
    q_emb = embed_model.encode([query])
    D, I = st.session_state.faiss_index.search(np.array(q_emb), k)
    return [st.session_state.vector_store_chunks[i] for i in I[0]]

def calculate_match_score(resume_text, job_desc):
    """Calculate similarity score between resume and job description"""
    resume_emb = embed_model.encode([resume_text])
    job_emb = embed_model.encode([job_desc])
    
    # Calculate cosine similarity
    similarity = np.dot(resume_emb, job_emb.T)[0][0]
    return float(similarity)

def optimize_resume(resume_text, job_description):
    """Optimize resume based on job description using LLM"""
    relevant_chunks = retrieve_chunks(job_description, k=6)
    context = "\n".join(relevant_chunks)
    
    prompt = f"""
    You are an expert resume writer and career coach. Optimize the following resume to better match the job description.
    
    Job Description:
    {job_description}
    
    Current Resume:
    {resume_text}
    
    Relevant Resume Sections:
    {context}
    
    Please:
    1. Rewrite the resume to highlight skills and experiences that match the job requirements
    2. Use industry-specific keywords from the job description
    3. Quantify achievements where possible
    4. Maintain professional tone and formatting
    5. Focus on the most relevant experiences
    
    Provide the optimized resume in a clear, professional format.
    Also list the key improvements made.
    """
    
    return query_ollama(prompt)

def generate_interview_questions(job_description, resume_text):
    """Generate role-specific interview questions based on job description and resume"""
    prompt = f"""
    You are an experienced hiring manager and technical interviewer. Generate comprehensive interview questions based on the following:
    
    Job Description:
    {job_description}
    
    Candidate Resume:
    {resume_text}
    
    Please generate:
    1. 5-7 behavioral questions (STAR method)
    2. 5-7 technical/skills-based questions
    3. 3-5 situational questions
    4. 2-3 questions about experience gaps or concerns
    
    Categorize each question type and provide brief guidance on what to look for in answers.
    Format the response clearly with category headers.
    """
    
    return query_ollama(prompt)

# Main UI
st.title("📄 Resume Optimizer & Interview Question Generator")
st.markdown("Leverage AI to optimize your resume and prepare for interviews")

# Sidebar
with st.sidebar:
    st.header("🔧 Configuration")
    
    # Check Ollama connection
    if check_ollama_connection():
        st.success(f"✅ Connected to Ollama ({MODEL_NAME})")
    else:
        st.error("❌ Ollama is not running. Start it with: `ollama serve`")
    
    # Resume upload status
    st.subheader("📑 Resume Status")
    if st.session_state.resume_text:
        st.success(f"✅ Resume loaded ({len(st.session_state.resume_text)} characters)")
        st.metric("Chunks Created", len(st.session_state.vector_store_chunks) if st.session_state.vector_store_chunks else 0)
    else:
        st.info("⏳ No resume uploaded yet")

# Main content tabs
tab4 = st.tabs(["📊 Resume Analysis"])[0]


# Tab 4: Full Analysis
with tab4:
    st.header("Complete Analysis")
    st.write("Upload a resume and enter a job description for a complete analysis in one go.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 Upload Resume")
        uploaded_file_full = st.file_uploader(
            "Choose your resume file",
            type=['pdf', 'docx', 'txt'],
            key="resume_uploader_full"
        )
    
    with col2:
        st.subheader("📝 Job Description")
        job_description_full = st.text_area(
            "Enter the job description:",
            placeholder="Paste the full job description here...",
            height=300,
            key="job_desc_full"
        )
    
    if st.button("Run Full Analysis", key="full_analysis_btn"):
        if uploaded_file_full is None:
            st.warning("Please upload a resume.")
        elif not job_description_full.strip():
            st.warning("Please enter a job description.")
        else:
            with st.spinner("Running complete analysis..."):
                # Extract and process resume
                resume_text_full = extract_text_from_file(uploaded_file_full)
                
                if resume_text_full:
                    build_vector_store(resume_text_full)
                    
                    # Calculate match score
                    match_score_full = calculate_match_score(resume_text_full, job_description_full)
                    
                    # Optimize resume
                    optimized_resume_full = optimize_resume(resume_text_full, job_description_full)
                    
                    # Generate questions
                    interview_questions_full = generate_interview_questions(job_description_full, resume_text_full)
                    
                    if optimized_resume_full and interview_questions_full:
                        # Display results
                        st.success("✅ Analysis complete! Scroll down to see your results.")
                        st.markdown("---")
                        
                        # Match score with better visualization
                        st.markdown("## 📊 Resume-Job Match Analysis")
                        col_score1, col_score2, col_score3 = st.columns([1,2,1])
                        with col_score2:
                            # Color code the score
                            score_color = "🟢" if match_score_full > 0.7 else "🟡" if match_score_full > 0.5 else "🔴"
                            st.metric(f"{score_color} Match Score", f"{match_score_full:.1%}", delta="Higher is better")
                        
                        st.markdown("---")
                        
                        # Optimized resume section
                        st.subheader("📋 Optimized Resume")
                        with st.expander("Click to view optimized resume", expanded=True):
                            st.markdown("### Your AI-Optimized Resume:")
                            st.markdown("---")
                            # Display as formatted text
                            st.markdown(f"```\n{optimized_resume_full}\n```")
                            
                            # Also provide as editable text area
                            st.markdown("**Edit your resume below:**")
                            edited_resume = st.text_area(
                                "You can edit the optimized resume:",
                                value=optimized_resume_full,
                                height=400,
                                key="editable_resume"
                            )
                        
                        # Interview questions section
                        st.subheader("❓ Interview Questions")
                        with st.expander("Click to view interview questions", expanded=True):
                            st.markdown("### Role-Specific Interview Questions:")
                            st.markdown("---")
                            # Display questions with better formatting
                            st.markdown(f"```\n{interview_questions_full}\n```")
                        
                        # Download buttons
                        col_dl1, col_dl2 = st.columns(2)
                        with col_dl1:
                            st.download_button(
                                label="📥 Download Optimized Resume",
                                data=edited_resume,  # Use edited version if user made changes
                                file_name="optimized_resume.txt",
                                mime="text/plain",
                                key="dl_resume_full"
                            )
                        with col_dl2:
                            st.download_button(
                                label="📥 Download Interview Questions",
                                data=interview_questions_full,
                                file_name="interview_questions.txt",
                                mime="text/plain",
                                key="dl_questions_full"
                            )

# Footer
st.divider()
st.markdown("""
---
**Resume Optimizer & Interview Generator** | Powered by LLaMA 2 (via Ollama) and Sentence Transformers
""")

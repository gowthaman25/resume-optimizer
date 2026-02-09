import streamlit as st
import numpy as np
import io
import docx
import PyPDF2
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from numpy.linalg import norm

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Resume Optimizer (Cloud)",
    page_icon="📄",
    layout="wide"
)

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_embed_model()

# ---------------- UTIL FUNCTIONS ----------------
def read_docx(file_bytes):
    doc = docx.Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs)

def read_pdf(file_bytes):
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    return "\n".join(p.extract_text() or "" for p in reader.pages)

def extract_text(uploaded_file):
    data = uploaded_file.read()
    if uploaded_file.name.endswith(".pdf"):
        return read_pdf(data)
    if uploaded_file.name.endswith(".docx"):
        return read_docx(data)
    if uploaded_file.name.endswith(".txt"):
        return data.decode("utf-8")
    return ""

def cosine_similarity(a, b):
    return float(np.dot(a, b.T) / (norm(a) * norm(b)))

def query_llm(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content

# ---------------- AI TASKS ----------------
def optimize_resume(resume, jd):
    prompt = f"""
You are an expert resume writer.

Job Description:
{jd}

Resume:
{resume}

Rewrite the resume to:
- Match job keywords
- Highlight relevant experience
- Quantify achievements
- Keep professional formatting

Also list key improvements.
"""
    return query_llm(prompt)

def generate_questions(resume, jd):
    prompt = f"""
Generate interview questions based on:

Job Description:
{jd}

Resume:
{resume}

Include:
- Behavioral questions
- Technical questions
- Situational questions
"""
    return query_llm(prompt)

# ---------------- UI ----------------
st.title("📄 Resume Optimizer & Interview Prep (Cloud)")

uploaded_file = st.file_uploader(
    "Upload your resume (PDF / DOCX / TXT)",
    type=["pdf", "docx", "txt"]
)

job_description = st.text_area(
    "Paste Job Description",
    height=300
)

if st.button("Run Analysis"):
    if not uploaded_file or not job_description.strip():
        st.warning("Please upload a resume and provide job description.")
    else:
        with st.spinner("Analyzing..."):
            resume_text = extract_text(uploaded_file)

            # Embeddings
            resume_emb = embed_model.encode([resume_text])[0]
            jd_emb = embed_model.encode([job_description])[0]

            score = cosine_similarity(resume_emb, jd_emb)

            optimized_resume = optimize_resume(resume_text, job_description)
            interview_questions = generate_questions(resume_text, job_description)

        st.success("Analysis complete!")

        st.metric("Resume–JD Match Score", f"{score:.2%}")

        st.subheader("📋 Optimized Resume")
        st.text_area("Optimized Resume", optimized_resume, height=400)

        st.subheader("❓ Interview Questions")
        st.text_area("Interview Questions", interview_questions, height=350)

        st.download_button(
            "Download Optimized Resume",
            optimized_resume,
            file_name="optimized_resume.txt"
        )

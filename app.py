import streamlit as st
import pdfplumber
import pandas as pd
import re
import spacy
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer


nlp = spacy.load("en_core_web_sm")

job_roles = ["Data Scientist", "Data Analyst", "Software Engineer", "ML Engineer", "AI Researcher"]
model = RandomForestClassifier(n_estimators=100, random_state=42)


job_skills = ["Python", "Machine Learning", "SQL", "Deep Learning", "Power BI", "TensorFlow"]


def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])


def extract_name(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return None


def extract_email(text):
    match = re.findall(r"[a-zA-Z0-9+_.-]+@[a-zA-Z0-9.-]+", text)
    return match[0] if match else None


def extract_phone(text):
    match = re.findall(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4,}", text)
    return match[0] if match else None


DEGREES = ["B.Sc", "BSc", "M.Sc", "MSc", "PhD", "MBA", "B.Tech", "M.Tech"]
def extract_education(text):
    return [degree for degree in DEGREES if degree in text] or None

TECH_SKILLS = ["Python", "Machine Learning", "SQL", "Deep Learning", "Power BI", "TensorFlow", "AWS"]
def extract_skills(text):
    return [skill for skill in TECH_SKILLS if skill.lower() in text.lower()] or None


def infer_job_role(skills):
    if skills is None:
        return "Unknown"
    skills = set(skills)
    if "Machine Learning" in skills or "Deep Learning" in skills:
        return "Data Scientist"
    if "SQL" in skills and "Power BI" in skills:
        return "Data Analyst"
    if "Python" in skills and "TensorFlow" in skills:
        return "ML Engineer"
    if "AWS" in skills:
        return "Software Engineer"
    return "Unknown"


def calculate_resume_score(skills):
    return len(set(skills) & set(job_skills)) * 10 if skills else 0

st.set_page_config(page_title="AI Resume Screener", layout="wide")
st.title("📄 AI Resume Screening System")

uploaded_file = st.file_uploader("📤 Upload a Resume (PDF)", type=["pdf"])
if uploaded_file:
    st.subheader("📑 Extracted Resume Details")
    
    
    text = extract_text_from_pdf(uploaded_file)

    
    name = extract_name(text)
    email = extract_email(text)
    phone = extract_phone(text)
    education = extract_education(text)
    skills = extract_skills(text)
    job_role = infer_job_role(skills)
    resume_score = calculate_resume_score(skills)

    
    st.write(f"**👤 Name:** {name or 'Not Found'}")
    st.write(f"**📧 Email:** {email or 'Not Found'}")
    st.write(f"**📞 Phone:** {phone or 'Not Found'}")
    st.write(f"**🎓 Education:** {', '.join(education) if education else 'Not Found'}")
    st.write(f"**💡 Skills:** {', '.join(skills) if skills else 'Not Found'}")
    st.write(f"**🔮 Predicted Job Role:** {job_role}")
    st.write(f"**📊 Resume Score:** {resume_score}/100")

    
    if "resumes" not in st.session_state:
        st.session_state.resumes = []
    
    st.session_state.resumes.append({
        "Filename": uploaded_file.name,
        "Name": name,
        "Email": email,
        "Phone": phone,
        "Education": education,
        "Skills": skills,
        "Job Role": job_role,
        "Resume Score": resume_score
    })


if "resumes" in st.session_state and len(st.session_state.resumes) > 0:
    st.subheader("📋 All Uploaded Resumes")
    df = pd.DataFrame()

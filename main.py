import os
import tempfile
import json
import re
from typing import List, Optional
from datetime import datetime

from flask import Flask, request, jsonify, abort
from flask_cors import CORS
from werkzeug.utils import secure_filename

import PyPDF2
from groq import Groq

import pytesseract
from pdf2image import convert_from_path
from PIL import Image, ImageFilter
import cv2
import numpy as np

from docx import Document
from pptx import Presentation

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"], supports_credentials=True)

# Initialize Groq client
try:
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    groq_client = Groq(api_key=groq_api_key)
except Exception as e:
    print(f"Error initializing Groq client: {e}")
    groq_client = None

SUPPORTED_EXTENSIONS = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation"
}

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

def allowed_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in SUPPORTED_EXTENSIONS

def validate_file(file_storage):
    filename = file_storage.filename
    if not filename:
        return False, "No filename provided"
    if not allowed_file(filename):
        return False, f"Unsupported file type: {os.path.splitext(filename)[1].lower()}"
    file_storage.seek(0, os.SEEK_END)
    size = file_storage.tell()
    file_storage.seek(0)
    if size == 0:
        return False, "File is empty"
    if size > MAX_FILE_SIZE:
        return False, "File size too large (max 10MB)"
    return True, None

def preprocess_image_pil(image: Image.Image) -> Image.Image:
    image = image.convert("L")
    image = image.point(lambda x: 0 if x < 140 else 255, '1')
    image = image.filter(ImageFilter.SHARPEN)
    return image

def preprocess_image_cv2(np_img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    norm_img = np.zeros((gray.shape[0], gray.shape[1]))
    img = cv2.normalize(gray, norm_img, 0, 255, cv2.NORM_MINMAX)
    img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)[1]
    img = cv2.GaussianBlur(img, (1, 1), 0)
    return img

def ocr_image_file(image_path: str) -> str:
    try:
        image = Image.open(image_path)
        image = preprocess_image_pil(image)
        text = pytesseract.image_to_string(image)
        if text.strip():
            return text
        np_img = cv2.imread(image_path)
        img = preprocess_image_cv2(np_img)
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        print(f"OCR image extraction failed: {e}")
        return ""

def ocr_extract_text_from_pdf(pdf_path: str) -> str:
    try:
        pages = convert_from_path(pdf_path, dpi=300)
        texts = []
        for page in pages:
            pre_img = preprocess_image_pil(page)
            page_text = pytesseract.image_to_string(pre_img)
            texts.append(page_text)
        return "\n".join(texts).strip()
    except Exception as e:
        print(f"OCR extraction failed: {e}")
        return ""

def extract_text_from_pdf_optimized(file_path: str) -> str:
    text = ""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        text = text.strip()
        if text:
            return text
    except Exception as e:
        print(f"Error reading PDF normally: {e}")
    ocr_text = ocr_extract_text_from_pdf(file_path)
    if ocr_text:
        return ocr_text
    else:
        return ""

def extract_text_from_docx(file_path: str) -> str:
    try:
        doc = Document(file_path)
        full_text = []
        for paragraph in doc.paragraphs:
            full_text.append(paragraph.text)
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    full_text.append(cell.text)
        return '\n'.join(full_text).strip()
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        return ""

def extract_text_from_pptx(file_path: str) -> str:
    try:
        prs = Presentation(file_path)
        full_text = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    full_text.append(shape.text)
        return '\n'.join(full_text).strip()
    except Exception as e:
        print(f"Error extracting text from PPTX: {e}")
        return ""

def extract_text_from_file(file_path: str, file_extension: str) -> str:
    if file_extension == ".pdf":
        return extract_text_from_pdf_optimized(file_path)
    elif file_extension == ".docx":
        return extract_text_from_docx(file_path)
    elif file_extension == ".pptx":
        return extract_text_from_pptx(file_path)
    elif file_extension in [".png", ".jpg", ".jpeg"]:
        return ocr_image_file(file_path)
    else:
        return ""

def extract_json_from_response(response_text: str) -> dict:
    match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if match:
        json_str = match.group(0)
        return json.loads(json_str)
    return json.loads(response_text)

# Experience extraction helpers
def parse_date(text):
    for fmt in ("%b %Y", "%B %Y", "%m/%Y", "%Y"):
        try:
            return datetime.strptime(text, fmt)
        except Exception:
            continue
    return None

def extract_periods(resume_text):
    patterns = [
        r'([A-Za-z]{3,9} \d{4})\s*[-to]+\s*(Present|[A-Za-z]{3,9} \d{4})',
        r'(\d{4})\s*[-to]+\s*(Present|\d{4})',
        r'(\d{2}/\d{4})\s*[-to]+\s*(Present|\d{2}/\d{4})'
    ]
    periods = []
    for pat in patterns:
        for m in re.findall(pat, resume_text, re.IGNORECASE):
            start, end = m
            start = start.replace("to", "-").replace("TO", "-").replace("To", "-").strip()
            end = end.strip()
            periods.append((start, end))
    return periods

def calculate_years_from_periods(periods):
    total_months = 0
    now = datetime.now()
    for start, end in periods:
        s = parse_date(start) or (parse_date("Jan "+start) if re.match(r"\d{4}$", start) else None)
        if end.lower() == "present":
            e = now
        else:
            e = parse_date(end) or (parse_date("Jan "+end) if re.match(r"\d{4}$", end) else None)
        if s and e and e > s:
            diff = (e.year - s.year) * 12 + (e.month - s.month)
            total_months += max(0, diff)
    return round(total_months / 12, 2) if total_months else None

def extract_resume_experience(resume_text: str) -> Optional[float]:
    patterns = [
        r'(\d{1,2}(?:\.\d+)?)\s*\+?\s*years? of experience',
        r'over\s*(\d{1,2}(?:\.\d+)?)\s*years',
        r'(\d{1,2}(?:\.\d+)?)\s*\+?\s*years? experience',
        r'(\d{1,2}(?:\.\d+)?)\s*\+?\s*years?'
    ]
    for pat in patterns:
        m = re.search(pat, resume_text, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                continue
    periods = extract_periods(resume_text)
    years = calculate_years_from_periods(periods)
    return years

def extract_required_experience(jd_text: str) -> tuple:
    patterns = [
        r'(\d{1,2})\s*[-to]{0,3}\s*(\d{1,2})\s*years',
        r'(?:minimum|min\.?)\s*(\d{1,2})\s*years',
        r'at\s*least\s*(\d{1,2})\s*years',
        r'(\d{1,2})\+?\s*years'
    ]
    for pat in patterns:
        m = re.search(pat, jd_text, re.IGNORECASE)
        if m:
            if len(m.groups()) == 2:
                return int(m.group(1)), int(m.group(2))
            elif len(m.groups()) == 1:
                return int(m.group(1)), None
    return None, None

def extract_skills(text):
    skill_keywords = [
        'python', 'java', 'c++', 'sql', 'excel', 'communication', 'leadership', 'project management',
        'aws', 'azure', 'docker', 'kubernetes', 'javascript', 'react', 'node', 'django', 'flask',
        'machine learning', 'data analysis', 'cloud', 'git', 'linux', 'powerpoint', 'word', 'rest', 'api'
    ]
    found = set()
    text_lower = text.lower()
    for skill in skill_keywords:
        if skill in text_lower:
            found.add(skill)
    return found

def classify_and_summarize_document(text: str, filename: str) -> dict:
    if not groq_client:
        abort(500, description="Groq client not initialized. Please check your API key.")
    cleaned_text = re.sub(r'\s+', ' ', text).strip()
    limited_text = cleaned_text[:3000] if len(cleaned_text) > 3000 else cleaned_text
    prompt = f"""
    Analyze the following document text from file "{filename}" and provide:
    1. Document Type: Classify this document as one of these types:
       - Offer Letter
       - Experience Letter
       - Medical Certificate
       - Leave Letter
       - Resignation Letter
       - Salary Certificate
       - Presentation/Slides (for PowerPoint files)
       - Report/Document (for Word documents)
       - Other (specify what type)
    2. Summary: Provide a one-line summary of the document's main content
    3. Confidence: Rate your confidence in the classification (High/Medium/Low)
    Please respond ONLY in valid JSON format with keys: "document_type", "summary", "confidence"
    Document Text:
    {limited_text}
    """
    try:
        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert document classifier. Analyze documents and provide accurate classifications and summaries in valid JSON format only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=500
        )
        response_text = response.choices[0].message.content.strip()
        try:
            result = extract_json_from_response(response_text)
            return {
                "document_type": result.get("document_type", "Unknown"),
                "summary": result.get("summary", "Unable to generate summary"),
                "confidence": result.get("confidence", "Low")
            }
        except Exception as json_error:
            print(f"JSON parsing error: {json_error}")
            print(f"Response text: {response_text}")
            doc_type = "Unknown"
            summary = "Unable to generate summary"
            confidence = "Low"
            type_match = re.search(r'"?document_type"?\s*:\s*"([^"]*)"', response_text, re.IGNORECASE)
            summary_match = re.search(r'"?summary"?\s*:\s*"([^"]*)"', response_text, re.IGNORECASE)
            confidence_match = re.search(r'"?confidence"?\s*:\s*"([^"]*)"', response_text, re.IGNORECASE)
            if type_match:
                doc_type = type_match.group(1)
            if summary_match:
                summary = summary_match.group(1)
            if confidence_match:
                confidence = confidence_match.group(1)
            return {
                "document_type": doc_type,
                "summary": summary,
                "confidence": confidence
            }
    except Exception as e:
        print(f"Groq API error: {e}")
        abort(500, description=f"Error with Groq API: {str(e)}")

def score_resume_against_jd(jd_text: str, resume_text: str, resume_filename: str) -> dict:
    if not groq_client:
        abort(500, description="Groq client not initialized. Please check your API key.")

    min_exp, max_exp = extract_required_experience(jd_text)
    cand_exp = extract_resume_experience(resume_text)

    exp_score = 0
    exp_reason = ""
    exp_match = False
    tolerance = 0.5

    if min_exp is not None:
        if cand_exp is None:
            exp_score = 10
            exp_reason = "Could not determine candidate's experience from resume."
        elif cand_exp < min_exp - tolerance:
            exp_score = max(0, int(50 * (cand_exp / min_exp)))
            exp_reason = f"Candidate has {cand_exp} years, required is at least {min_exp}. Underqualified."
        elif max_exp and cand_exp > max_exp + tolerance:
            exp_score = max(10, int(50 * (max_exp / cand_exp)))
            exp_reason = f"Candidate has {cand_exp} years, required is {min_exp}-{max_exp}. Overqualified."
        else:
            exp_score = 50
            exp_reason = f"Candidate experience ({cand_exp} years) matches requirement ({min_exp}{'-'+str(max_exp) if max_exp else ''})."
            exp_match = True
    else:
        exp_score = 40 if cand_exp else 10
        exp_reason = "No explicit experience requirement found in JD." if cand_exp else "Could not determine candidate's experience."

    jd_skills = extract_skills(jd_text)
    resume_skills = extract_skills(resume_text)
    matched_skills = jd_skills & resume_skills
    missing_skills = jd_skills - resume_skills
    total_skills = len(jd_skills)
    if total_skills == 0:
        skills_score = 40
        skills_reason = "No explicit skills found in JD."
    else:
        skills_score = int(50 * (len(matched_skills) / total_skills))
        skills_reason = f"{len(matched_skills)}/{total_skills} required skills matched."

    total_score = exp_score + skills_score
    qualification_status = "Qualified" if exp_score >= 30 and skills_score >= 30 else "Underqualified"

    return {
        "score": total_score,
        "qualification_status": qualification_status,
        "summary": f"Experience: {exp_reason} | Skills: {skills_reason}",
        "experience_match": exp_match,
        "experience_comment": exp_reason,
        "skills_matched": ", ".join(sorted(matched_skills)),
        "skills_missing": ", ".join(sorted(missing_skills)),
        "strengths": f"- Experience Score: {exp_score}\n- Skills Score: {skills_score}",
        "weaknesses": skills_reason if skills_score < 30 else ""
    }

# Routes

@app.route("/score-resumes", methods=["POST"])
def score_resumes():
    jd_file = request.files.get("jd_file")
    jd_text = request.form.get("jd_text", "").strip()
    resumes = request.files.getlist("resumes")

    if not jd_file and not jd_text:
        return jsonify({"detail": "No job description provided (file or text required)"}), 400

    if not resumes:
        return jsonify({"detail": "No resumes uploaded"}), 400

    jd_text_extracted = ""
    temp_jd_path = None

    if jd_file:
        valid, err = validate_file(jd_file)
        if not valid:
            return jsonify({"detail": f"JD file error: {err}"}), 400
        filename = secure_filename(jd_file.filename)
        ext = os.path.splitext(filename)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            return jsonify({"detail": f"Unsupported JD file type: {ext}"}), 400
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            jd_file.save(temp_file.name)
            temp_jd_path = temp_file.name
        jd_text_extracted = extract_text_from_file(temp_jd_path, ext)
        if temp_jd_path and os.path.exists(temp_jd_path):
            try:
                os.unlink(temp_jd_path)
            except Exception:
                pass
    else:
        jd_text_extracted = jd_text

    if not jd_text_extracted or len(jd_text_extracted) < 10:
        return jsonify({"detail": "No usable text in job description"}), 400

    results = []
    failed_files = []

    for resume in resumes:
        if not resume.filename:
            failed_files.append("Unknown filename - No filename provided")
            continue
        valid, err = validate_file(resume)
        if not valid:
            failed_files.append(f"{resume.filename} - {err}")
            continue
        filename = secure_filename(resume.filename)
        ext = os.path.splitext(filename)[1].lower()
        temp_resume_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
                resume.save(temp_file.name)
                temp_resume_path = temp_file.name
            resume_text = extract_text_from_file(temp_resume_path, ext)
            if not resume_text or len(resume_text.strip()) < 10:
                failed_files.append(f"{resume.filename} - No readable text extracted")
                continue
            result = score_resume_against_jd(jd_text_extracted, resume_text, resume.filename)
            results.append({
                "filename": resume.filename,
                "score": result["score"],
                "qualification_status": result["qualification_status"],
                "summary": result["summary"],
                "experience_match": result["experience_match"],
                "experience_comment": result["experience_comment"],
                "skills_matched": result["skills_matched"],
                "skills_missing": result["skills_missing"],
                "strengths": result["strengths"],
                "weaknesses": result["weaknesses"]
            })
        except Exception as e:
            failed_files.append(f"{resume.filename} - Error: {str(e)}")
        finally:
            if temp_resume_path and os.path.exists(temp_resume_path):
                try:
                    os.unlink(temp_resume_path)
                except Exception:
                    pass

    return jsonify({
        "job_description": jd_text_extracted[:2000],
        "results": results,
        "failed_files": failed_files
    })

@app.route("/analyze-file", methods=["POST"])
def analyze_single_file():
    if "file" not in request.files:
        return jsonify({"detail": "No file uploaded"}), 400
    file = request.files["file"]
    valid, err = validate_file(file)
    if not valid:
        return jsonify({"detail": err}), 400

    filename = secure_filename(file.filename)
    ext = os.path.splitext(filename)[1].lower()
    temp_file_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            file.save(temp_file.name)
            temp_file_path = temp_file.name
        extracted_text = extract_text_from_file(temp_file_path, ext)
        if not extracted_text.strip():
            return jsonify({"detail": "No text could be extracted from the file."}), 400
        if len(extracted_text.strip()) < 10:
            return jsonify({"detail": "Insufficient text content in the file for analysis."}), 400
        analysis_result = classify_and_summarize_document(extracted_text, filename)
        return jsonify({
            "filename": filename,
            "document_type": analysis_result["document_type"],
            "summary": analysis_result["summary"],
            "confidence": analysis_result["confidence"],
            "extracted_text_length": len(extracted_text)
        })
    except Exception as e:
        print(f"Unexpected error in analyze_single_file: {e}")
        return jsonify({"detail": f"An unexpected error occurred: {str(e)}"}), 500
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass

@app.route("/analyze-multiple-files", methods=["POST"])
def analyze_multiple_files():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"detail": "No files uploaded"}), 400
    if len(files) > 10:
        return jsonify({"detail": "Too many files. Maximum 10 files per request."}), 400

    analyses = []
    failed_files = []
    successful_count = 0

    for file in files:
        if not file.filename:
            failed_files.append("Unknown filename - No filename provided")
            continue
        valid, err = validate_file(file)
        if not valid:
            failed_files.append(f"{file.filename} - {err}")
            continue
        filename = secure_filename(file.filename)
        ext = os.path.splitext(filename)[1].lower()
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
                file.save(temp_file.name)
                temp_file_path = temp_file.name
            extracted_text = extract_text_from_file(temp_file_path, ext)
            if not extracted_text.strip():
                failed_files.append(f"{file.filename} - No text could be extracted")
                continue
            if len(extracted_text.strip()) < 10:
                failed_files.append(f"{file.filename} - Insufficient text content")
                continue
            analysis_result = classify_and_summarize_document(extracted_text, filename)
            analyses.append({
                "filename": filename,
                "document_type": analysis_result["document_type"],
                "summary": analysis_result["summary"],
                "confidence": analysis_result["confidence"],
                "extracted_text_length": len(extracted_text)
            })
            successful_count += 1
        except Exception as e:
            failed_files.append(f"{file.filename} - Processing error: {str(e)}")
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception:
                    pass

    return jsonify({
        "total_files": len(files),
        "successful_analyses": successful_count,
        "failed_files": failed_files,
        "analyses": analyses
    })

@app.route("/health", methods=["GET"])
def health_check():
    groq_status = "connected" if groq_client else "not connected"
    api_key_status = "present" if os.getenv("GROQ_API_KEY") else "missing"
    return jsonify({
        "status": "healthy",
        "message": "Document Analyzer API is running",
        "groq_status": groq_status,
        "api_key_status": api_key_status,
        "supported_file_types": list(SUPPORTED_EXTENSIONS.keys())
    })

@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "message": "Document & Image Analyzer API",
        "version": "2.1.0",
        "supported_file_types": list(SUPPORTED_EXTENSIONS.keys()),
        "endpoints": {
            "analyze_single": "/analyze-file",
            "analyze_multiple": "/analyze-multiple-files",
            "score_resumes": "/score-resumes",
            "health": "/health"
        }
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)

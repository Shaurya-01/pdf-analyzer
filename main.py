import os
import tempfile
import json
import re
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import PyPDF2
from groq import Groq

# OCR and image processing
import pytesseract
from pdf2image import convert_from_path
from PIL import Image, ImageFilter
import cv2
import numpy as np

# Document processing libraries
from docx import Document
from pptx import Presentation

# Load environment variables
load_dotenv()

app = FastAPI(title="Document & Image Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq client
try:
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    groq_client = Groq(api_key=groq_api_key)
except Exception as e:
    print(f"Error initializing Groq client: {e}")
    groq_client = None

class FileAnalysis(BaseModel):
    filename: str
    document_type: str
    summary: str
    confidence: str
    extracted_text_length: int

class MultiFileAnalysis(BaseModel):
    total_files: int
    successful_analyses: int
    failed_files: List[str]
    analyses: List[FileAnalysis]

# Supported file types and their MIME types
SUPPORTED_EXTENSIONS = {
    ".pdf": "application/pdf",
    ".png": "image/png", 
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation"
}

def validate_file_type(file: UploadFile) -> bool:
    """Validate file type based on extension and MIME type."""
    if not file.filename:
        return False
    
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return False
    
    # Check MIME type if available
    expected_mime = SUPPORTED_EXTENSIONS[ext]
    if file.content_type and file.content_type != expected_mime:
        # Some flexibility for image types
        if ext in [".jpg", ".jpeg"] and file.content_type in ["image/jpeg", "image/jpg"]:
            return True
        return False
    
    return True

# --- OCR Utilities ---

def preprocess_image_pil(image: Image.Image) -> Image.Image:
    """Preprocess a PIL image for better OCR."""
    # Convert to grayscale
    image = image.convert("L")
    # Apply thresholding
    image = image.point(lambda x: 0 if x < 140 else 255, '1')
    # Sharpen
    image = image.filter(ImageFilter.SHARPEN)
    return image

def preprocess_image_cv2(np_img: np.ndarray) -> np.ndarray:
    """Preprocess a numpy image (OpenCV) for better OCR."""
    gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    norm_img = np.zeros((gray.shape[0], gray.shape[1]))
    img = cv2.normalize(gray, norm_img, 0, 255, cv2.NORM_MINMAX)
    img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)[1]
    img = cv2.GaussianBlur(img, (1, 1), 0)
    return img

def ocr_image_file(image_path: str) -> str:
    """Extract text from an image file using pytesseract with preprocessing."""
    try:
        # Try PIL preprocessing
        image = Image.open(image_path)
        image = preprocess_image_pil(image)
        text = pytesseract.image_to_string(image)
        if text.strip():
            return text
        # If PIL fails, try OpenCV preprocessing
        np_img = cv2.imread(image_path)
        img = preprocess_image_cv2(np_img)
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        print(f"OCR image extraction failed: {e}")
        return ""

def ocr_extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF using OCR on each page (with preprocessing)."""
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
    """Try normal extraction, fallback to OCR if needed."""
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

    # Fallback to OCR
    print("Falling back to OCR extraction for PDF")
    ocr_text = ocr_extract_text_from_pdf(file_path)
    if ocr_text:
        return ocr_text
    else:
        return ""

def extract_text_from_docx(file_path: str) -> str:
    """Extract text from a DOCX file."""
    try:
        doc = Document(file_path)
        full_text = []
        
        # Extract text from paragraphs
        for paragraph in doc.paragraphs:
            full_text.append(paragraph.text)
        
        # Extract text from tables
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    full_text.append(cell.text)
        
        return '\n'.join(full_text).strip()
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        return ""

def extract_text_from_pptx(file_path: str) -> str:
    """Extract text from a PPTX file."""
    try:
        prs = Presentation(file_path)
        full_text = []
        
        # Extract text from each slide
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    full_text.append(shape.text)
        
        return '\n'.join(full_text).strip()
    except Exception as e:
        print(f"Error extracting text from PPTX: {e}")
        return ""

def extract_text_from_file(file_path: str, file_extension: str) -> str:
    """Extract text from file based on its extension."""
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

# --- LLM Utilities ---

def extract_json_from_response(response_text: str) -> dict:
    """Extract JSON from a string that may contain markdown code blocks."""
    if "```json" in response_text:
        start_idx = response_text.find("```json") + len("```json")
        end_idx = response_text.find("```", start_idx)
        if end_idx != -1:
            json_str = response_text[start_idx:end_idx].strip()
        else:
            json_str = response_text
    elif "```" in response_text:
        start_idx = response_text.find("```") + len("```")
        end_idx = response_text.find("```", start_idx)
        if end_idx != -1:
            json_str = response_text[start_idx:end_idx].strip()
        else:
            json_str = response_text
    else:
        json_str = response_text
    return json.loads(json_str)

def classify_and_summarize_document(text: str, filename: str) -> dict:
    """Use Groq API with LLaMA to classify document type and summarize."""
    if not groq_client:
        raise HTTPException(status_code=500, detail="Groq client not initialized. Please check your API key.")

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
            # Fallback: try to extract values with regex
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
        raise HTTPException(status_code=500, detail=f"Error with Groq API: {str(e)}")

# --- Main Endpoints ---

@app.post("/analyze-file", response_model=FileAnalysis)
async def analyze_single_file(file: UploadFile = File(...)):
    """Analyze a single PDF, DOCX, PPTX, or image file."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    if not validate_file_type(file):
        ext = os.path.splitext(file.filename)[1].lower()
        supported_exts = ", ".join(SUPPORTED_EXTENSIONS.keys())
        raise HTTPException(
            status_code=400, 
            detail=f"File type '{ext}' not supported. Supported types: {supported_exts}"
        )

    try:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        if len(content) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=400, detail="File size too large. Maximum size is 10MB.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading uploaded file: {str(e)}")

    ext = os.path.splitext(file.filename)[1].lower()
    temp_file_path = None
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name

        # Extract text based on file type
        extracted_text = extract_text_from_file(temp_file_path, ext)

        if not extracted_text.strip():
            raise HTTPException(
                status_code=400, 
                detail="No text could be extracted from the file. The file might be corrupted or contain no readable text."
            )
        
        if len(extracted_text.strip()) < 10:
            raise HTTPException(
                status_code=400, 
                detail="Insufficient text content in the file for analysis."
            )

        analysis_result = classify_and_summarize_document(extracted_text, file.filename)
        
        return FileAnalysis(
            filename=file.filename,
            document_type=analysis_result["document_type"],
            summary=analysis_result["summary"],
            confidence=analysis_result["confidence"],
            extracted_text_length=len(extracted_text)
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in analyze_single_file: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as cleanup_error:
                print(f"Error cleaning up temporary file: {cleanup_error}")

@app.post("/analyze-multiple-files", response_model=MultiFileAnalysis)
async def analyze_multiple_files(files: List[UploadFile] = File(...)):
    """Analyze multiple files of supported types."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    if len(files) > 10:  # Limit to 10 files per request
        raise HTTPException(status_code=400, detail="Too many files. Maximum 10 files per request.")

    analyses = []
    failed_files = []
    successful_count = 0

    for file in files:
        try:
            if not file.filename:
                failed_files.append(f"Unknown filename - No filename provided")
                continue
                
            if not validate_file_type(file):
                ext = os.path.splitext(file.filename)[1].lower() if file.filename else "unknown"
                supported_exts = ", ".join(SUPPORTED_EXTENSIONS.keys())
                failed_files.append(f"{file.filename} - Unsupported file type '{ext}'. Supported: {supported_exts}")
                continue

            content = await file.read()
            if len(content) == 0:
                failed_files.append(f"{file.filename} - Empty file")
                continue
                
            if len(content) > 10 * 1024 * 1024:  # 10MB limit per file
                failed_files.append(f"{file.filename} - File too large (>10MB)")
                continue

            ext = os.path.splitext(file.filename)[1].lower()
            temp_file_path = None
            
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
                    temp_file.write(content)
                    temp_file_path = temp_file.name

                # Extract text based on file type
                extracted_text = extract_text_from_file(temp_file_path, ext)

                if not extracted_text.strip():
                    failed_files.append(f"{file.filename} - No text could be extracted")
                    continue
                    
                if len(extracted_text.strip()) < 10:
                    failed_files.append(f"{file.filename} - Insufficient text content")
                    continue

                analysis_result = classify_and_summarize_document(extracted_text, file.filename)
                
                analyses.append(FileAnalysis(
                    filename=file.filename,
                    document_type=analysis_result["document_type"],
                    summary=analysis_result["summary"],
                    confidence=analysis_result["confidence"],
                    extracted_text_length=len(extracted_text)
                ))
                
                successful_count += 1

            except Exception as e:
                failed_files.append(f"{file.filename} - Processing error: {str(e)}")
            finally:
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.unlink(temp_file_path)
                    except Exception as cleanup_error:
                        print(f"Error cleaning up temporary file {temp_file_path}: {cleanup_error}")

        except Exception as e:
            failed_files.append(f"{file.filename if file.filename else 'Unknown'} - Error: {str(e)}")

    return MultiFileAnalysis(
        total_files=len(files),
        successful_analyses=successful_count,
        failed_files=failed_files,
        analyses=analyses
    )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    groq_status = "connected" if groq_client else "not connected"
    api_key_status = "present" if os.getenv("GROQ_API_KEY") else "missing"
    return {
        "status": "healthy",
        "message": "Document Analyzer API is running",
        "groq_status": groq_status,
        "api_key_status": api_key_status,
        "supported_file_types": list(SUPPORTED_EXTENSIONS.keys())
    }

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Document & Image Analyzer API",
        "version": "2.0.0",
        "supported_file_types": list(SUPPORTED_EXTENSIONS.keys()),
        "endpoints": {
            "analyze_single": "/analyze-file",
            "analyze_multiple": "/analyze-multiple-files",
            "health": "/health",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

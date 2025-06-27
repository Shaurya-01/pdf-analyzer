import os
import tempfile
import json
import re

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

class DocumentAnalysis(BaseModel):
    document_type: str
    summary: str
    confidence: str

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

def classify_and_summarize_document(text: str) -> DocumentAnalysis:
    """Use Groq API with LLaMA to classify document type and summarize."""
    if not groq_client:
        raise HTTPException(status_code=500, detail="Groq client not initialized. Please check your API key.")

    cleaned_text = re.sub(r'\s+', ' ', text).strip()
    limited_text = cleaned_text[:3000] if len(cleaned_text) > 3000 else cleaned_text

    prompt = f"""
    Analyze the following document text and provide:
    1. Document Type: Classify this document as one of these types:
       - Offer Letter
       - Experience Letter
       - Medical Certificate
       - Leave Letter
       - Resignation Letter
       - Salary Certificate
       - Other (specify what type)
    2. Summary: Provide a one-line summary of the document's main content
    3. Confidence: Rate your confidence in the classification (High/Medium/Low)
    Please respond ONLY in valid JSON format with keys: "document_type", "summary", "confidence"
    Document Text:
    {limited_text}
    """

    try:
        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",  # Best available model
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
            return DocumentAnalysis(
                document_type=result.get("document_type", "Unknown"),
                summary=result.get("summary", "Unable to generate summary"),
                confidence=result.get("confidence", "Low")
            )
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
            return DocumentAnalysis(
                document_type=doc_type,
                summary=summary,
                confidence=confidence
            )
    except Exception as e:
        print(f"Groq API error: {e}")
        raise HTTPException(status_code=500, detail=f"Error with Groq API: {str(e)}")

# --- Main Endpoint ---

@app.post("/analyze-file", response_model=DocumentAnalysis)
async def analyze_file(file: UploadFile = File(...)):
    """Analyze PDF or image file, extract text, classify, and summarize."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".pdf", ".png", ".jpg", ".jpeg"]:
        raise HTTPException(status_code=400, detail="Only PDF, PNG, JPG, and JPEG files are allowed")
    try:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File size too large. Maximum size is 10MB.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading uploaded file: {str(e)}")
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name

        # Extract text based on file type
        if ext == ".pdf":
            extracted_text = extract_text_from_pdf_optimized(temp_file_path)
        else:
            extracted_text = ocr_image_file(temp_file_path)

        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the file, even with OCR. The file might be corrupted or contain no readable text.")
        if len(extracted_text.strip()) < 10:
            raise HTTPException(status_code=400, detail="Insufficient text content in the file for analysis.")
        analysis = classify_and_summarize_document(extracted_text)
        return analysis
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in analyze_file: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as cleanup_error:
                print(f"Error cleaning up temporary file: {cleanup_error}")

@app.get("/health")
async def health_check():
    groq_status = "connected" if groq_client else "not connected"
    api_key_status = "present" if os.getenv("GROQ_API_KEY") else "missing"
    return {
        "status": "healthy",
        "message": "Document Analyzer API is running",
        "groq_status": groq_status,
        "api_key_status": api_key_status
    }

@app.get("/")
async def root():
    return {
        "message": "Document & Image Analyzer API",
        "version": "1.0.0",
        "endpoints": {
            "analyze": "/analyze-file",
            "health": "/health",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
import os
import cv2
import numpy as np
from PIL import Image
import pytesseract
import pdfplumber

def extract_text_from_pdf(file_path):
    try:
        with pdfplumber.open(file_path) as pdf:
            return pdf.pages[0].extract_text()
    except Exception as e:
        print(f"Error reading PDF: {str(e)}")
        return ""

def preprocess_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to read image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    kernel = np.ones((1, 1), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    processed = cv2.convertScaleAbs(processed, alpha=1.5, beta=0)
    return processed

def extract_text_from_image(image_path):
    processed_img = preprocess_image(image_path)
    pil_img = Image.fromarray(processed_img)
    custom_config = r'--oem 3 --psm 6'
    return pytesseract.image_to_string(pil_img, config=custom_config)

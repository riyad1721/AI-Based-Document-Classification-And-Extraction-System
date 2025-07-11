from sentence_transformers import SentenceTransformer, util
from text_extraction import extract_text_from_pdf, extract_text_from_image
import os
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
# print(hf_token, type(hf_token))
if hf_token is None:
    raise ValueError("HF_TOKEN not found in environment variables.")

model = SentenceTransformer("all-MiniLM-L6-v2", use_auth_token=hf_token)

reference_texts = {
    "Invoice": "This document is an invoice showing vendor name, invoice number, total amount and date.",
    "Bank Statement": "This is a bank statement listing transactions, account number, dates and balances.",
    "Money Receipt": "This is a money receipt showing a receipt number, total amount paid in cash, VAT breakdown,Tax,cash receipt, items sold, and payer contact details."
}

def classify_document_type(file_path):
    if file_path.lower().endswith('.pdf'):
        text = extract_text_from_pdf(file_path)
    elif file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        text = extract_text_from_image(file_path)
    else:
        return None
    doc_embedding = model.encode(text, convert_to_tensor=True)
    scores = {}
    for label, ref_text in reference_texts.items():
        ref_embedding = model.encode(ref_text, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(doc_embedding, ref_embedding)
        scores[label] = similarity.item()
    predicted_label = max(scores, key=scores.get)
    return predicted_label

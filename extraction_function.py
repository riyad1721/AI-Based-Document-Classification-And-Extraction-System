import re
import pandas as pd
from text_extraction import extract_text_from_pdf, extract_text_from_image

def extract_bank_statement_data(file_path):
    transactions = []
    text = extract_text_from_pdf(file_path)

    account_match = re.search(r'Account Number / Name / Currency Code:\s*([\d-]+)', text)
    account_number = account_match.group(1) if account_match else "N/A"

    transaction_lines = re.finditer(
        r'(?P<date>\d{2}-[A-Za-z]{3}-\d{4})\s+'
        r'(?P<type>.+?)\s+'
        r'(?P<description>.*?)\s+'
        r'(?:[\d,]+\.\d{2}\s+){0,2}'
        r'(?P<balance>-?[\d,]+\.\d{2})',
        text
    )

    for match in transaction_lines:
        transactions.append({
            'Account Number': account_number,
            'Transaction Date': match.group('date'),
            'Description': f"{match.group('type')} , {match.group('description')}".strip(),
            'Amount': match.group('balance').replace(',', '')
        })

    return pd.DataFrame(transactions)


def extract_invoice_data(file_path):
    text = extract_text_from_pdf(file_path)
    normalized_text = ' '.join(text.split())

    inv_num_match = re.search(r'INVOICE\s+(\d+)', normalized_text)
    date_match = re.search(r'DATE\s+(\d{2}/\d{2}/\d{4})', normalized_text)
    subtotal_match = re.search(r'SUBTOTAL\s+([\d,]+\.\d{2})', normalized_text)
    vat_match = re.search(r'VAT TOTAL\s+([\d,]+\.\d{2})', normalized_text)

    total_amount = "N/A"
    if subtotal_match and vat_match:
        subtotal = float(subtotal_match.group(1).replace(',', ''))
        vat = float(vat_match.group(1).replace(',', ''))
        total_amount = f"{subtotal + vat:.2f}"
    else:
        total_match = re.search(r'(?:TOTAL|BALANCE DUE)\s+(?:GBP\s*)?([\d,]+\.\d{2})', normalized_text)
        if total_match:
            total_amount = total_match.group(1).replace(',', '')

    vendor_match = re.search(r'INVOICE TO(?:.*?INVOICE \d+)?\s+(.*?)\s+DATE', normalized_text)
    invoice_number = inv_num_match.group(1) if inv_num_match else "N/A"
    date = date_match.group(1) if date_match else "N/A"
    vendor_name = "N/A"
    if vendor_match:
        vendor_name = vendor_match.group(1).strip()
        vendor_name = re.sub(r'INVOICE\s+\d+|\d{2}/\d{2}/\d{4}', '', vendor_name).strip()

    return pd.DataFrame([{
        'Invoice Number': invoice_number,
        'Date': date,
        'Total Amount': total_amount,
        'Vendor Name': vendor_name
    }])


def extract_receipt_data(image_path):
    text = extract_text_from_image(image_path)

    receipt_num = "N/A"
    rf_match = re.search(r'RF RECEIPT No[:;]\s*([\w\s]+)', text)
    if rf_match:
        receipt_num = re.sub(r'[^\d\s]', '', rf_match.group(1)).strip()
        receipt_num = ' '.join(receipt_num.split())
    else:
        receipt_match = re.search(r'RECEIPT NO[:]?\s*((?:\d\s*){8,12})', text, re.IGNORECASE)
        if receipt_match:
            receipt_num = re.sub(r'\s', '', receipt_match.group(1))
        else:
            near_rf_match = re.search(r'(?:RF|RECEIPT).*?(\d{3}\s*\d\s*\d{3}\s*\d{3})', text, re.IGNORECASE)
            if near_rf_match:
                receipt_num = re.sub(r'\s', '', near_rf_match.group(1))

    date_match = re.search(r'RF (\d{2}/\d{2}/\d{4})', text) or \
                 re.search(r'DATE[:]?\s*(\d{2}/\d{2}/\d{4})', text) or \
                 re.search(r'(\d{2}/\d{2}/\d{4})\s*RF', text)
    date = date_match.group(1) if date_match else "N/A"

    cash_match = re.search(r'CASH\s*[\D]*([\d.]+)', text)
    change_match = re.search(r'CHANGE\s*[\D]*([\d.]+)', text)
    payment_amount = "N/A"
    if cash_match and change_match:
        try:
            payment_amount = f"{float(cash_match.group(1)) - float(change_match.group(1)):.2f}"
        except ValueError:
            pass

    payer_match = re.search(r'^(.*?)\n\s*SELF\s*EMPLOYED', text, re.MULTILINE | re.IGNORECASE)
    payer_name = payer_match.group(1).strip() if payer_match else "N/A"
    payer_name = re.sub(r'[^a-zA-Z\s]', '', payer_name).strip()

    return pd.DataFrame([{
        'Receipt Number': receipt_num,
        'Date': date,
        'Payment Amount': payment_amount,
        'Payer Name': payer_name
    }])
import os
import json
import argparse
from tqdm import tqdm
import jieba
import pdfplumber
import logging
import pytesseract
from PIL import Image
import pickle  # 用於保存處理後的數據

pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
# 設置日志記錄
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to read PDF and split into chunks with OCR fallback
def read_pdf(pdf_loc, page_infos: list = None):
    try:
        pdf = pdfplumber.open(pdf_loc)
    except Exception as e:
        logging.error(f"Error opening PDF file {pdf_loc}: {e}")
        return []
    
    try:
        pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
    except IndexError:
        logging.warning(f"Page range {page_infos} out of bounds for file {pdf_loc}. Extracting all pages.")
        pages = pdf.pages
    
    pdf_text = ''
    for page_number, page in enumerate(pages, start=1):
        try:
            text = page.extract_text()
            if text:
                logging.info(f"Extracted {len(text)} characters from page {page_number} of {pdf_loc} using pdfplumber.")
                pdf_text += text + "\n\n"
            else:
                # 如果pdfplumber無法提取文本，嘗試使用OCR
                logging.info(f"No text found on page {page_number} of {pdf_loc}. Attempting OCR.")
                # 將頁面轉換為圖像
                image = page.to_image(resolution=300).original
                # 使用PIL將圖像轉換為RGB模式
                pil_image = image.convert("RGB")
                # 使用pytesseract進行OCR
                ocr_text = pytesseract.image_to_string(pil_image, lang='chi_tra')  # 使用繁體中文語言包
                if ocr_text.strip():
                    logging.info(f"Extracted {len(ocr_text)} characters from page {page_number} of {pdf_loc} using OCR.")
                    pdf_text += ocr_text + "\n\n"
                else:
                    logging.warning(f"OCR failed to extract text from page {page_number} of {pdf_loc}.")
        except Exception as e:
            logging.error(f"Error processing page {page_number} in {pdf_loc}: {e}")
    pdf.close()
    
    splits = pdf_text.split('\n\n')
    splits = [split.strip() for split in splits if split.strip()]
    return splits

# Function to load data from source path
def load_data(source_path):
    masked_file_ls = os.listdir(source_path)
    corpus_dict = {}
    all_documents = []
    all_doc_ids = []
    missing_pdfs = []
    for file in tqdm(masked_file_ls, desc=f"Loading data from {source_path}"):
        try:
            file_id = int(file.replace('.pdf', ''))
        except ValueError:
            logging.warning(f"Skipping non-PDF or improperly named file: {file}")
            continue
        file_path = os.path.join(source_path, file)
        splits = read_pdf(file_path)
        if not splits:
            logging.warning(f"No content extracted from file: {file_path}")
            missing_pdfs.append(file)
        corpus_dict[file_id] = splits
        all_documents.extend(splits)
        all_doc_ids.extend([file_id] * len(splits))
    if missing_pdfs:
        logging.info(f"Total missing PDFs: {len(missing_pdfs)}")
        for pdf in missing_pdfs:
            logging.info(f"Missing PDF: {pdf}")
    return corpus_dict, all_documents, all_doc_ids

def main(args):
    # Load reference data
    corpus_dict_finance, documents_finance, doc_ids_finance = load_data(os.path.join(args.source_path, 'finance'))
    corpus_dict_insurance, documents_insurance, doc_ids_insurance = load_data(os.path.join(args.source_path, 'insurance'))

    # Save processed data
    with open(os.path.join(args.output_path, 'finance_data.pkl'), 'wb') as f_finance:
        pickle.dump((corpus_dict_finance, documents_finance, doc_ids_finance), f_finance)
    logging.info("Finance data saved.")

    with open(os.path.join(args.output_path, 'insurance_data.pkl'), 'wb') as f_insurance:
        pickle.dump((corpus_dict_insurance, documents_insurance, doc_ids_insurance), f_insurance)
    logging.info("Insurance data saved.")

    # Load FAQ mapping and split
    try:
        with open(os.path.join(args.source_path, 'faq/pid_map_content.json'), 'rb') as f_s:
            key_to_source_dict = json.load(f_s)  # 讀取參考資料文件
            key_to_source_dict = {int(key): value for key, value in key_to_source_dict.items()}
        logging.info("Loaded FAQ mapping.")
    except Exception as e:
        logging.error(f"Error reading FAQ mapping file: {e}")
        key_to_source_dict = {}
        faq_documents = []
        faq_doc_ids = []

    # Prepare FAQ documents
    faq_documents = []
    faq_doc_ids = []
    for key, value in key_to_source_dict.items():
        for q in value:
            combined = f"問題：{q['question']} 答案：{' '.join(q['answers'])}"
            faq_documents.append(combined)
            faq_doc_ids.append(key)

    # Save FAQ data
    with open(os.path.join(args.output_path, 'faq_data.pkl'), 'wb') as f_faq:
        pickle.dump((key_to_source_dict, faq_documents, faq_doc_ids), f_faq)
    logging.info("FAQ data saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Preprocessing Script for PDFs and FAQs.')
    parser.add_argument('--source_path', type=str, required=True, help='Path to the source data directory.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the processed data.')

    args = parser.parse_args()

    main(args)
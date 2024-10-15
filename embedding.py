# compute_embeddings.py
import os
import json
import argparse
from tqdm import tqdm
import pdfplumber
import logging
import pytesseract
from PIL import Image
from sentence_transformers import SentenceTransformer
import numpy as np

pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

# 設置日志記錄
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to read PDF and split into chunks with OCR fallback
def read_pdf(pdf_loc, page_infos: list = None):
    try:
        with pdfplumber.open(pdf_loc) as pdf:
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

            splits = pdf_text.split('\n\n')
            splits = [split.strip() for split in splits if split.strip()]
            return splits
    except Exception as e:
        logging.error(f"Error opening PDF file {pdf_loc}: {e}")
        return []

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

    # Load FAQ mapping and split
    try:
        with open(os.path.join(args.source_path, 'faq/pid_map_content.json'), 'r', encoding='utf8') as f_s:
            key_to_source_dict = json.load(f_s)  # 讀取參考資料文件
            key_to_source_dict = {int(key): value for key, value in key_to_source_dict.items()}
        logging.info("Loaded FAQ mapping.")
    except Exception as e:
        logging.error(f"Error reading FAQ mapping file: {e}")
        key_to_source_dict = {}
        faq_documents = []
        faq_doc_ids = []

    # 檢查是否包含所有需要的 FAQ doc_id
    required_faq_doc_ids = set()
    for q in args.questions.get('questions', []):
        if 101 <= int(q.get('qid', 0)) <= 150 and q.get('category') == 'faq':
            required_faq_doc_ids.update(q.get('source', []))

    missing_faq_doc_ids = [doc_id for doc_id in required_faq_doc_ids if doc_id not in key_to_source_dict]
    if missing_faq_doc_ids:
        logging.warning(f"The following FAQ doc_ids are missing in 'faq/pid_map_content.json': {missing_faq_doc_ids}")
    else:
        logging.info("All required FAQ doc_ids are present in 'faq/pid_map_content.json'.")

    # Prepare FAQ documents
    faq_documents = []
    faq_doc_ids = []
    for key, value in key_to_source_dict.items():
        for q in value:
            combined = f"問題：{q['question']} 答案：{' '.join(q['answers'])}"
            faq_documents.append(combined)
            faq_doc_ids.append(key)

    # Aggregate all documents
    all_documents = documents_finance + documents_insurance + faq_documents
    all_doc_ids = doc_ids_finance + doc_ids_insurance + faq_doc_ids

    # Load questions file
    try:
        with open(args.questions_path, 'r', encoding='utf8') as f:
            qs_ref = json.load(f)
        logging.info(f"Loaded questions from {args.questions_path}")
    except Exception as e:
        logging.error(f"Error reading question file {args.questions_path}: {e}")
        return

    # Initialize Embedding Model
    try:
        embedding_model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True, device=args.device)
        logging.info("Embedding model 'jinaai/jina-embeddings-v3' loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading embedding model: {e}")
        return

    # Compute Document Embeddings
    logging.info("Computing document embeddings...")
    try:
        document_embeddings = embedding_model.encode(
            all_documents,
            batch_size=32,  
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
            task="retrieval.passage"
        )
        logging.info(f"Computed embeddings for {len(all_documents)} documents.")
    except Exception as e:
        logging.error(f"Error during document embedding computation: {e}")
        return

    # Save Embeddings
    try:
        np.save(args.output_path, document_embeddings)
        logging.info(f"Document embeddings saved to {args.output_path}")
    except Exception as e:
        logging.error(f"Error saving embeddings to {args.output_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute and save document embeddings.')
    parser.add_argument('--questions_path', type=str, required=True, help='Path to the questions JSON file.')
    parser.add_argument('--source_path', type=str, required=True, help='Path to the source data directory.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the embeddings (.npy file).')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the embedding model on (cpu, cuda, mps).')
    
    args = parser.parse_args()

    # Load questions file for checking required FAQ doc_ids
    try:
        with open(args.questions_path, 'r', encoding='utf8') as f:
            args.questions = json.load(f)
        logging.info(f"Loaded questions from {args.questions_path} for embedding computation.")
    except Exception as e:
        logging.error(f"Error reading question file {args.questions_path}: {e}")
        args.questions = {}

    main(args)
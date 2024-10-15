# build_faiss_index.py

import os
import json
import argparse
from tqdm import tqdm
import pickle
import logging
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

import torch

# 設置日志記錄
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(args):
    # 加載預處理數據
    with open(os.path.join(args.processed_data_path, 'finance_data.pkl'), 'rb') as f_finance:
        corpus_dict_finance, documents_finance, doc_ids_finance = pickle.load(f_finance)
    logging.info("Loaded preprocessed finance data.")

    with open(os.path.join(args.processed_data_path, 'insurance_data.pkl'), 'rb') as f_insurance:
        corpus_dict_insurance, documents_insurance, doc_ids_insurance = pickle.load(f_insurance)
    logging.info("Loaded preprocessed insurance data.")

    with open(os.path.join(args.processed_data_path, 'faq_data.pkl'), 'rb') as f_faq:
        key_to_source_dict, faq_documents, faq_doc_ids = pickle.load(f_faq)
    logging.info("Loaded preprocessed FAQ data.")

    # 整合所有文檔
    all_documents = documents_finance + documents_insurance + faq_documents
    all_doc_ids = doc_ids_finance + doc_ids_insurance + faq_doc_ids

    # 初始化嵌入模型
    model_name = 'maidalun1020/bce-embedding-base_v1'  # 您的嵌入模型名稱
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_kwargs = {'device': device}
    encode_kwargs = {'batch_size': 64, 'normalize_embeddings': True}  # 移除 'show_progress_bar'

    embed_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    logging.info(f"Embedding model '{model_name}' initialized.")

    # 構建 FAISS 索引
    logging.info("Building FAISS index...")
    faiss_vectorstore = FAISS.from_texts(
        texts=all_documents,
        embedding=embed_model,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        metadatas=[{"source": doc_id} for doc_id in all_doc_ids]  # 添加元數據
    )
    logging.info("FAISS index built successfully.")

    # 保存 FAISS 索引
    faiss_index_path = os.path.join(args.output_path, 'faiss_index')
    os.makedirs(faiss_index_path, exist_ok=True)
    faiss_vectorstore.save_local(faiss_index_path)
    logging.info(f"FAISS index saved to {faiss_index_path}.")

    # 保存文檔 ID 映射（可選，因為已經在 metadatas 中保存）
    with open(os.path.join(args.output_path, 'doc_id_mapping.pkl'), 'wb') as f_mapping:
        pickle.dump(all_doc_ids, f_mapping)
    logging.info("Document ID mapping saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FAISS Index Building Script.')
    parser.add_argument('--processed_data_path', type=str, required=True, help='Path to the preprocessed data directory.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the FAISS index and mappings.')

    args = parser.parse_args()

    main(args)
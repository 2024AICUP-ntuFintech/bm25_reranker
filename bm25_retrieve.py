# import os
# import json
# import argparse

# from tqdm import tqdm
# import jieba  # 用於中文文本分詞
# import pdfplumber  # 用於從PDF文件中提取文字的工具
# from rank_bm25 import BM25Okapi  # 使用BM25演算法進行文件檢索


# # 載入參考資料，返回一個字典，key為檔案名稱，value為PDF檔內容的文本
# def load_data(source_path):
#     masked_file_ls = os.listdir(source_path)  # 獲取資料夾中的檔案列表
#     corpus_dict = {int(file.replace('.pdf', '')): read_pdf(os.path.join(source_path, file)) for file in tqdm(masked_file_ls)}  # 讀取每個PDF文件的文本，並以檔案名作為鍵，文本內容作為值存入字典
#     return corpus_dict


# # 讀取單個PDF文件並返回其文本內容
# def read_pdf(pdf_loc, page_infos: list = None):
#     pdf = pdfplumber.open(pdf_loc)  # 打開指定的PDF文件

#     # TODO: 可自行用其他方法讀入資料，或是對pdf中多模態資料（表格,圖片等）進行處理

#     # 如果指定了頁面範圍，則只提取該範圍的頁面，否則提取所有頁面
#     pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
#     pdf_text = ''
#     for _, page in enumerate(pages):  # 迴圈遍歷每一頁
#         text = page.extract_text()  # 提取頁面的文本內容
#         if text:
#             pdf_text += text
#     pdf.close()  # 關閉PDF文件

#     return pdf_text  # 返回萃取出的文本


# # 根據查詢語句和指定的來源，檢索答案
# def BM25_retrieve(qs, source, corpus_dict):
#     filtered_corpus = [corpus_dict[int(file)] for file in source]

#     # [TODO] 可自行替換其他檢索方式，以提升效能

#     tokenized_corpus = [list(jieba.cut_for_search(doc)) for doc in filtered_corpus]  # 將每篇文檔進行分詞
#     bm25 = BM25Okapi(tokenized_corpus)  # 使用BM25演算法建立檢索模型
#     tokenized_query = list(jieba.cut_for_search(qs))  # 將查詢語句進行分詞
#     ans = bm25.get_top_n(tokenized_query, list(filtered_corpus), n=1)  # 根據查詢語句檢索，返回最相關的文檔，其中n為可調整項
#     a = ans[0]
#     # 找回與最佳匹配文本相對應的檔案名
#     res = [key for key, value in corpus_dict.items() if value == a]
#     return res[0]  # 回傳檔案名


# if __name__ == "__main__":
#     # 使用argparse解析命令列參數
#     parser = argparse.ArgumentParser(description='Process some paths and files.')
#     parser.add_argument('--question_path', type=str, required=True, help='讀取發布題目路徑')  # 問題文件的路徑
#     parser.add_argument('--source_path', type=str, required=True, help='讀取參考資料路徑')  # 參考資料的路徑
#     parser.add_argument('--output_path', type=str, required=True, help='輸出符合參賽格式的答案路徑')  # 答案輸出的路徑

#     args = parser.parse_args()  # 解析參數

#     answer_dict = {"answers": []}  # 初始化字典

#     with open(args.question_path, 'rb') as f:
#         qs_ref = json.load(f)  # 讀取問題檔案

#     source_path_insurance = os.path.join(args.source_path, 'insurance')  # 設定參考資料路徑
#     corpus_dict_insurance = load_data(source_path_insurance)

#     source_path_finance = os.path.join(args.source_path, 'finance')  # 設定參考資料路徑
#     corpus_dict_finance = load_data(source_path_finance)

#     with open(os.path.join(args.source_path, 'faq/pid_map_content.json'), 'rb') as f_s:
#         key_to_source_dict = json.load(f_s)  # 讀取參考資料文件
#         key_to_source_dict = {int(key): value for key, value in key_to_source_dict.items()}

#     for q_dict in qs_ref['questions']:
#         if q_dict['category'] == 'finance':
#             # 進行檢索
#             retrieved = BM25_retrieve(q_dict['query'], q_dict['source'], corpus_dict_finance)
#             # 將結果加入字典
#             answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})

#         elif q_dict['category'] == 'insurance':
#             retrieved = BM25_retrieve(q_dict['query'], q_dict['source'], corpus_dict_insurance)
#             answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})

#         elif q_dict['category'] == 'faq':
#             corpus_dict_faq = {key: str(value) for key, value in key_to_source_dict.items() if key in q_dict['source']}
#             retrieved = BM25_retrieve(q_dict['query'], q_dict['source'], corpus_dict_faq)
#             answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})

#         else:
#             raise ValueError("Something went wrong")  # 如果過程有問題，拋出錯誤

#     # 將答案字典保存為json文件
#     with open(args.output_path, 'w', encoding='utf8') as f:
#         json.dump(answer_dict, f, ensure_ascii=False, indent=4)  # 儲存檔案，確保格式和非ASCII字符

#------------------------------------------------------------------------------------------------------------------------
# import os
# import json
# import argparse

# from tqdm import tqdm
# import pdfplumber  # 用於從 PDF 文件中提取文字
# import jieba  # 用於中文文本分詞
# from bert_score import score  # 使用 BERTScore 進行相似度計算


# def read_pdf(pdf_loc, page_infos: list = None):
#     """
#     從單個 PDF 文件中提取文本內容。

#     Args:
#         pdf_loc: PDF 文件的路徑。
#         page_infos: 可選，頁面範圍 [start, end)。

#     Returns:
#         pdf_text: 提取出的文本內容。
#     """
#     with pdfplumber.open(pdf_loc) as pdf:
#         pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
#         pdf_text = ''
#         for page in pages:
#             text = page.extract_text()
#             if text:
#                 pdf_text += text
#     return pdf_text


# def load_data(source_path):
#     """
#     從指定資料夾中載入所有 PDF 文件的文本內容。

#     Args:
#         source_path: 資料夾的路徑。

#     Returns:
#         corpus_dict: 字典，鍵為文件ID，值為文件文本。
#     """
#     masked_file_ls = os.listdir(source_path)
#     corpus_dict = {}
#     for file in tqdm(masked_file_ls, desc=f"載入資料自 {source_path}"):
#         try:
#             file_id = int(file.replace('.pdf', ''))
#             file_path = os.path.join(source_path, file)
#             corpus_dict[file_id] = read_pdf(file_path)
#         except ValueError:
#             print(f"跳過無效的文件：{file}")
#     return corpus_dict


# def split_text(text, max_length=500, overlap=50):
#     """
#     將長文本分割成較小的段落，帶有重疊區域以保持上下文連貫。

#     Args:
#         text: 待分割的文本。
#         max_length: 每個段落的最大長度（以字符計）。
#         overlap: 段落之間的重疊字符數。

#     Returns:
#         segments: 分割後的文本段落列表。
#     """
#     segments = []
#     start = 0
#     text_length = len(text)
#     while start < text_length:
#         end = start + max_length
#         segment = text[start:end]
#         segments.append(segment)
#         start = end - overlap  # 重疊部分
#     return segments


# def compute_bertscore(query, document_segments, lang='zh', model_type='bert-base-chinese'):
#     """
#     計算查詢與文檔段落之間的 BERTScore，返回最高得分。

#     Args:
#         query: 查詢問題。
#         document_segments: 文檔分割後的段落列表。
#         lang: 語言代碼，預設為中文 'zh'。
#         model_type: BERTScore 使用的模型類型。

#     Returns:
#         max_f1: 文檔中最高的 F1 得分。
#     """
#     max_f1 = 0.0
#     for segment in document_segments:
#         # 計算 BERTScore
#         P, R, F1 = score([query], [segment], lang=lang, model_type=model_type, verbose=False)
#         current_f1 = F1.mean().item()
#         if current_f1 > max_f1:
#             max_f1 = current_f1
#     return max_f1


# def retrieve_best_source(query, source_ids, corpus_dict, max_length=500, overlap=50):
#     """
#     為給定的查詢，從候選來源中選擇最相關的文檔。

#     Args:
#         query: 查詢問題。
#         source_ids: 候選文檔ID列表。
#         corpus_dict: 文檔ID到文本的映射字典。
#         max_length: 文檔分段的最大長度。
#         overlap: 分段時的重疊字符數。

#     Returns:
#         best_source_id: 最相關的文檔ID。
#         best_score: 該文檔的最高得分。
#     """
#     best_score = -1.0
#     best_source_id = None

#     for source_id in source_ids:
#         if source_id not in corpus_dict:
#             print(f"來源ID {source_id} 不存在於資料庫中，跳過。")
#             continue

#         document = corpus_dict[source_id]
#         segments = split_text(document, max_length=max_length, overlap=overlap)

#         # 計算該文檔與查詢的最高 BERTScore F1
#         score_f1 = compute_bertscore(query, segments)

#         # 更新最佳來源
#         if score_f1 > best_score:
#             best_score = score_f1
#             best_source_id = source_id

#     return best_source_id, best_score


# def load_faq_data(faq_json_path):
#     """
#     從 FAQ JSON 文件中載入資料。

#     Args:
#         faq_json_path: FAQ JSON 文件的路徑。

#     Returns:
#         faq_dict: 字典，鍵為文件ID，值為文件文本或包含 'question' 和 'answers' 的列表。
#     """
#     with open(faq_json_path, 'r', encoding='utf8') as f:
#         key_to_source_dict = json.load(f)
#         key_to_source_dict = {int(key): value for key, value in key_to_source_dict.items()}
#     return key_to_source_dict


# def main(question_path, source_path, output_path):
#     """
#     主函數，處理問題並檢索最相關的文檔。

#     Args:
#         question_path: 問題 JSON 文件的路徑。
#         source_path: 參考資料的根路徑。
#         output_path: 輸出結果的 JSON 文件路徑。
#     """
#     answer_dict = {"answers": []}

#     # 讀取問題文件
#     with open(question_path, 'r', encoding='utf8') as f:
#         qs_ref = json.load(f)

#     # 加載 FAQ 資料
#     faq_json_path = os.path.join(source_path, 'faq', 'pid_map_content.json')
#     if os.path.exists(faq_json_path):
#         corpus_dict_faq = load_faq_data(faq_json_path)
#     else:
#         corpus_dict_faq = {}

#     # 加載並生成 'insurance' 類別的文檔
#     source_path_insurance = os.path.join(source_path, 'insurance')
#     corpus_dict_insurance = load_data(source_path_insurance)

#     # 加載並生成 'finance' 類別的文檔
#     source_path_finance = os.path.join(source_path, 'finance')
#     corpus_dict_finance = load_data(source_path_finance)

#     # 處理每個問題
#     for q_dict in tqdm(qs_ref['questions'], desc="處理問題"):
#         qid = q_dict.get('qid')
#         category = q_dict.get('category')
#         query = q_dict.get('query')
#         source_ids = q_dict.get('source')  # 來源文件ID列表

#         if category == 'finance':
#             best_source_id, best_score = retrieve_best_source(query, source_ids, corpus_dict_finance)
#         elif category == 'insurance':
#             best_source_id, best_score = retrieve_best_source(query, source_ids, corpus_dict_insurance)
#         elif category == 'faq':
#             # 將 FAQ 資料轉換為字符串
#             corpus_dict_faq_filtered = {key: str(value) for key, value in corpus_dict_faq.items() if key in source_ids}
#             best_source_id, best_score = retrieve_best_source(query, source_ids, corpus_dict_faq_filtered)
#         else:
#             print(f"未知的類別：{category}，跳過問題ID：{qid}")
#             best_source_id = None
#             best_score = None

#         # 將結果加入答案字典
#         answer_entry = {
#             "qid": qid,
#             "retrieve": best_source_id
#         }
#         if best_score is not None:
#             answer_entry["score"] = round(best_score, 4)  # 保留四位小數
#         answer_dict['answers'].append(answer_entry)

#     # 保存答案到 JSON 文件
#     with open(output_path, 'w', encoding='utf8') as f:
#         json.dump(answer_dict, f, ensure_ascii=False, indent=4)

#     print(f"檢索完成。結果已保存至 {output_path}")


# if __name__ == "__main__":
#     # 使用 argparse 解析命令列參數
#     parser = argparse.ArgumentParser(description='處理一些路徑和文件。')
#     parser.add_argument('--question_path', type=str, required=True, help='讀取發布題目路徑')  # 問題文件的路徑
#     parser.add_argument('--source_path', type=str, required=True, help='讀取參考資料路徑')  # 參考資料的路徑
#     parser.add_argument('--output_path', type=str, required=True, help='輸出符合參賽格式的答案路徑')  # 答案輸出的路徑

#     args = parser.parse_args()

#     main(args.question_path, args.source_path, args.output_path)

# python bm25_retrieve.py \
#     --question_path /Users/danny9199/Downloads/玉山AI競賽/競賽資料集/dataset/preliminary/questions_example.json \
#     --source_path /Users/danny9199/Downloads/玉山AI競賽/競賽資料集/reference \
#     --output_path /Users/danny9199/Downloads/玉山AI競賽/競賽資料集/dataset/preliminary/pred_retrieve.json

#------------------------------------------------------------------------------------------------------------------------
# import os
# import json
# import argparse

# from tqdm import tqdm
# import jieba  # 用于中文文本分词
# import pdfplumber  # 用于从PDF文件中提取文字的工具
# from rank_bm25 import BM25Okapi  # 使用BM25算法进行文件检索

# import torch
# from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline


# # 加载参考资料，返回一个字典，key为文件名，value为PDF文件内容的文本
# def load_data(source_path):
#     masked_file_ls = os.listdir(source_path)  # 获取文件夹中的文件列表
#     corpus_dict = {int(file.replace('.pdf', '')): read_pdf(os.path.join(source_path, file)) 
#                    for file in tqdm(masked_file_ls, desc=f"Loading data from {source_path}")}
#     return corpus_dict


# # 读取单个PDF文件并返回其文本内容
# def read_pdf(pdf_loc, page_infos: list = None):
#     pdf = pdfplumber.open(pdf_loc)  # 打开指定的PDF文件

#     # 如果指定了页面范围，则只提取该范围的页面，否则提取所有页面
#     pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
#     pdf_text = ''
#     for _, page in enumerate(pages):  # 循环遍历每一页
#         text = page.extract_text()  # 提取页面的文本内容
#         if text:
#             pdf_text += text
#     pdf.close()  # 关闭PDF文件

#     return pdf_text  # 返回提取出的文本


# # 根据查询语句和指定的来源，检索答案
# def BM25_retrieve(qs, source, corpus_dict, qa_pipeline):
#     # 构建 (doc_id, doc_text) 的列表
#     filtered_corpus = [(int(file), corpus_dict[int(file)]) for file in source]

#     # 对文档进行分词
#     tokenized_corpus = [list(jieba.cut_for_search(doc_text)) for doc_id, doc_text in filtered_corpus]
#     bm25 = BM25Okapi(tokenized_corpus)
#     tokenized_query = list(jieba.cut_for_search(qs))
    
#     # 获取文档文本
#     corpus_texts = [doc_text for doc_id, doc_text in filtered_corpus]

#     # 根据source的长度，决定返回的候选文档数量
#     if len(source) > 10:
#         top_n = 10
#     else:
#         top_n = 5

#     # 获取前top_n个候选文档
#     ans = bm25.get_top_n(tokenized_query, corpus_texts, n=top_n)

#     # 构建文档文本到ID的映射
#     doc_text_to_id = {doc_text: doc_id for doc_id, doc_text in filtered_corpus}
#     # 获取候选文档的ID列表
#     retrieved_doc_ids = [doc_text_to_id[doc_text] for doc_text in ans]

#     # 使用QA模型对每个候选文档进行评分
#     scores = []
#     print(f"\nQuery: {qs}")
#     for doc_id, doc_text in zip(retrieved_doc_ids, ans):
#         QA_input = {'question': qs, 'context': doc_text}
#         try:
#             result = qa_pipeline(QA_input)
#             score = result['score']
#             scores.append((doc_id, score))
#             print(f"Document ID: {doc_id}, Score: {score:.4f}")
#         except Exception as e:
#             print(f"Error processing Document ID: {doc_id}, Error: {e}")
#             scores.append((doc_id, 0))  # 如果出错，赋予最低分

#     # 选择得分最高的文档ID作为最终结果
#     if scores:
#         best_doc_id, best_score = max(scores, key=lambda x: x[1])
#         return best_doc_id
#     else:
#         return None  # 如果没有候选文档


# if __name__ == "__main__":
#     # 使用argparse解析命令行参数
#     parser = argparse.ArgumentParser(description='Process some paths and files.')
#     parser.add_argument('--question_path', type=str, required=True, help='读取发布题目路径')  # 问题文件的路径
#     parser.add_argument('--source_path', type=str, required=True, help='读取参考资料路径')  # 参考资料的路径
#     parser.add_argument('--output_path', type=str, required=True, help='输出符合参赛格式的答案路径')  # 答案输出的路径

#     args = parser.parse_args()  # 解析参数

#     answer_dict = {"answers": []}  # 初始化字典

#     # 读取问题文件
#     with open(args.question_path, 'rb') as f:
#         qs_ref = json.load(f)

#     # 加载参考资料
#     source_path_insurance = os.path.join(args.source_path, 'insurance')  # 设置参考资料路径
#     corpus_dict_insurance = load_data(source_path_insurance)

#     source_path_finance = os.path.join(args.source_path, 'finance')  # 设置参考资料路径
#     corpus_dict_finance = load_data(source_path_finance)

#     # 加载FAQ映射
#     with open(os.path.join(args.source_path, 'faq/pid_map_content.json'), 'rb') as f_s:
#         key_to_source_dict = json.load(f_s)
#         key_to_source_dict = {int(key): value for key, value in key_to_source_dict.items()}

#     # 加载QA模型和分词器
#     qa_model_name = 'liam168/qa-roberta-base-chinese-extractive'
#     qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
#     qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
#     qa_pipeline_instance = pipeline('question-answering', model=qa_model, tokenizer=qa_tokenizer)

#     # 如果使用GPU，请将模型移动到GPU
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     qa_model.to(device)

#     # 遍历所有问题
#     for q_dict in tqdm(qs_ref['questions'], desc="Processing questions"):
#         qid = q_dict.get('qid')
#         query = q_dict.get('query')
#         category = q_dict.get('category')
#         source = q_dict.get('source')

#         if not all([qid, query, category, source]):
#             print(f"Skipping incomplete question entry: {q_dict}")
#             continue

#         if category == 'finance':
#             # 进行检索
#             retrieved = BM25_retrieve(query, source, corpus_dict_finance, qa_pipeline_instance)
#             # 将结果加入字典
#             if retrieved is not None:
#                 answer_dict['answers'].append({"qid": qid, "retrieve": retrieved})
#             else:
#                 answer_dict['answers'].append({"qid": qid, "retrieve": None})

#         elif category == 'insurance':
#             retrieved = BM25_retrieve(query, source, corpus_dict_insurance, qa_pipeline_instance)
#             answer_dict['answers'].append({"qid": qid, "retrieve": retrieved})

#         elif category == 'faq':
#             corpus_dict_faq = {key: str(value) for key, value in key_to_source_dict.items() if key in source}
#             retrieved = BM25_retrieve(query, source, corpus_dict_faq, qa_pipeline_instance)
#             answer_dict['answers'].append({"qid": qid, "retrieve": retrieved})

#         else:
#             print(f"Unknown category '{category}' for QID: {qid}")
#             answer_dict['answers'].append({"qid": qid, "retrieve": None})

#     # 将答案字典保存为json文件
#     with open(args.output_path, 'w', encoding='utf8') as f:
#         json.dump(answer_dict, f, ensure_ascii=False, indent=4)  # 保存文件，确保格式和非ASCII字符

# python bm25_retrieve.py \
#     --question_path /Users/danny9199/Downloads/玉山AI競賽/競賽資料集/dataset/preliminary/questions_example.json \
#     --source_path /Users/danny9199/Downloads/玉山AI競賽/競賽資料集/reference \
#     --output_path /Users/danny9199/Downloads/玉山AI競賽/競賽資料集/dataset/preliminary/pred_retrieve.json

#------------------------------------------------------------------------------------------------------------------------
# import os
# import json
# import argparse
# from tqdm import tqdm
# import jieba
# import pdfplumber
# import torch
# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# from rank_bm25 import BM25Okapi
# import numpy as np
# import logging
# import pytesseract
# from PIL import Image

# # 設置日志記錄
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Function to read PDF and split into chunks with OCR fallback
# def read_pdf(pdf_loc, page_infos: list = None):
#     try:
#         pdf = pdfplumber.open(pdf_loc)
#     except Exception as e:
#         logging.error(f"Error opening PDF file {pdf_loc}: {e}")
#         return []
    
#     try:
#         pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
#     except IndexError:
#         logging.warning(f"Page range {page_infos} out of bounds for file {pdf_loc}. Extracting all pages.")
#         pages = pdf.pages
    
#     pdf_text = ''
#     for page_number, page in enumerate(pages, start=1):
#         try:
#             text = page.extract_text()
#             if text:
#                 logging.info(f"Extracted {len(text)} characters from page {page_number} of {pdf_loc} using pdfplumber.")
#                 pdf_text += text + "\n\n"
#             else:
#                 # 如果pdfplumber無法提取文本，嘗試使用OCR
#                 logging.info(f"No text found on page {page_number} of {pdf_loc}. Attempting OCR.")
#                 # 將頁面轉換為圖像
#                 image = page.to_image(resolution=300).original
#                 # 使用PIL將圖像轉換為RGB模式
#                 pil_image = image.convert("RGB")
#                 # 使用pytesseract進行OCR
#                 ocr_text = pytesseract.image_to_string(pil_image, lang='chi_tra')  # 使用繁體中文語言包
#                 if ocr_text.strip():
#                     logging.info(f"Extracted {len(ocr_text)} characters from page {page_number} of {pdf_loc} using OCR.")
#                     pdf_text += ocr_text + "\n\n"
#                 else:
#                     logging.warning(f"OCR failed to extract text from page {page_number} of {pdf_loc}.")
#         except Exception as e:
#             logging.error(f"Error processing page {page_number} in {pdf_loc}: {e}")
#     pdf.close()
    
#     splits = pdf_text.split('\n\n')
#     splits = [split.strip() for split in splits if split.strip()]
#     return splits

# # Function to load data from source path
# def load_data(source_path):
#     masked_file_ls = os.listdir(source_path)
#     corpus_dict = {}
#     all_documents = []
#     all_doc_ids = []
#     for file in tqdm(masked_file_ls, desc=f"Loading data from {source_path}"):
#         try:
#             file_id = int(file.replace('.pdf', ''))
#         except ValueError:
#             logging.warning(f"Skipping non-PDF or improperly named file: {file}")
#             continue
#         file_path = os.path.join(source_path, file)
#         splits = read_pdf(file_path)
#         if not splits:
#             logging.warning(f"No content extracted from file: {file_path}")
#         corpus_dict[file_id] = splits
#         all_documents.extend(splits)
#         all_doc_ids.extend([file_id] * len(splits))
#     return corpus_dict, all_documents, all_doc_ids

# # Function to tokenize documents for BM25
# def tokenize_corpus(corpus):
#     tokenized_corpus = [jieba.lcut(doc) for doc in corpus]
#     return tokenized_corpus

# # Function to initialize BM25
# def initialize_bm25(tokenized_corpus):
#     bm25 = BM25Okapi(tokenized_corpus)
#     return bm25

# # Function to load Reranker model
# def load_reranker(model_name, device):
#     try:
#         reranker = AutoModelForSequenceClassification.from_pretrained(model_name)
#         tokenizer = AutoTokenizer.from_pretrained(model_name)
#         reranker.to(device)
#         reranker.eval()
#         logging.info(f"Reranker model '{model_name}' loaded successfully.")
#         return reranker, tokenizer
#     except Exception as e:
#         logging.error(f"Error loading Reranker model {model_name}: {e}")
#         return None, None

# # Function to compute reranker scores using BAAI/bge-reranker-large
# def compute_reranker_scores(query, documents, reranker_model, reranker_tokenizer, device):
#     reranker_model.eval()
#     sentence_pairs = [[query, doc] for doc in documents]
#     scores = []
#     with torch.no_grad():
#         for i in tqdm(range(0, len(sentence_pairs), 32), desc="Computing Reranker Scores"):
#             batch = sentence_pairs[i:i+32]
#             inputs = reranker_tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
#             inputs = {k: v.to(device) for k, v in inputs.items()}
#             outputs = reranker_model(**inputs)
#             logits = outputs.logits.view(-1).float()
#             # 假設較高的 logits 表示較高的相關性
#             batch_scores = logits.cpu().numpy()
#             scores.extend(batch_scores)
#     return scores

# def main(args):
#     answer_dict = {"answers": []}

#     # Read questions file
#     try:
#         with open(args.question_path, 'r', encoding='utf8') as f:
#             qs_ref = json.load(f)
#         logging.info(f"Loaded questions from {args.question_path}")
#     except Exception as e:
#         logging.error(f"Error reading question file {args.question_path}: {e}")
#         return

#     # Load reference data
#     corpus_dict_finance, documents_finance, doc_ids_finance = load_data(os.path.join(args.source_path, 'finance'))
#     corpus_dict_insurance, documents_insurance, doc_ids_insurance = load_data(os.path.join(args.source_path, 'insurance'))

#     # Load FAQ mapping and split
#     try:
#         with open(os.path.join(args.source_path, 'faq/pid_map_content.json'), 'rb') as f_s:
#             key_to_source_dict = json.load(f_s)  # 讀取參考資料文件
#             key_to_source_dict = {int(key): value for key, value in key_to_source_dict.items()}
#         logging.info("Loaded FAQ mapping.")
#     except Exception as e:
#         logging.error(f"Error reading FAQ mapping file: {e}")
#         key_to_source_dict = {}
#         faq_documents = []
#         faq_doc_ids = []
    
#     # 檢查是否包含所有需要的 FAQ doc_id
#     required_faq_doc_ids = set()
#     for q in qs_ref.get('questions', []):
#         if 101 <= int(q.get('qid', 0)) <= 150 and q.get('category') == 'faq':
#             required_faq_doc_ids.update(q.get('source', []))
    
#     missing_faq_doc_ids = [doc_id for doc_id in required_faq_doc_ids if doc_id not in key_to_source_dict]
#     if missing_faq_doc_ids:
#         logging.warning(f"The following FAQ doc_ids are missing in 'faq/pid_map_content.json': {missing_faq_doc_ids}")
#     else:
#         logging.info("All required FAQ doc_ids are present in 'faq/pid_map_content.json'.")

#     # Prepare FAQ documents
#     faq_documents = []
#     faq_doc_ids = []
#     for key, value in key_to_source_dict.items():
#         for q in value:
#             combined = f"問題：{q['question']} 答案：{' '.join(q['answers'])}"
#             faq_documents.append(combined)
#             faq_doc_ids.append(key)

#     # Aggregate all documents
#     all_documents = documents_finance + documents_insurance + faq_documents
#     all_doc_ids = doc_ids_finance + doc_ids_insurance + faq_doc_ids

#     # Build a mapping from doc_id to list of indices in all_documents
#     doc_id_to_indices = {}
#     for idx, doc_id in enumerate(all_doc_ids):
#         if doc_id not in doc_id_to_indices:
#             doc_id_to_indices[doc_id] = []
#         doc_id_to_indices[doc_id].append(idx)

#     # Tokenize corpus for BM25
#     tokenized_corpus = tokenize_corpus(all_documents)

#     # Initialize BM25
#     bm25 = initialize_bm25(tokenized_corpus)
#     logging.info("BM25 index initialized.")

#     # Initialize Reranker model
#     reranker_model_name = 'BAAI/bge-reranker-large'  # Updated Reranker model
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     reranker_model, reranker_tokenizer = load_reranker(reranker_model_name, device)
#     if reranker_model is None:
#         logging.error("Reranker model failed to load. Exiting.")
#         return

#     # Log loaded doc_ids for verification
#     loaded_doc_ids = set(all_doc_ids)
#     logging.info(f"Total loaded doc_ids: {len(loaded_doc_ids)}")

#     # Process each question
#     for q_dict in tqdm(qs_ref.get('questions', []), desc="Processing questions"):
#         qid = q_dict.get('qid')
#         query = q_dict.get('query')
#         category = q_dict.get('category')
#         source = q_dict.get('source')

#         if not all([qid, query, category, source]):
#             logging.warning(f"Skipping incomplete question entry: {q_dict}")
#             answer_dict['answers'].append({"qid": qid, "retrieve": None})
#             continue

#         # Select the appropriate corpus
#         if category == 'finance':
#             corpus_dict = corpus_dict_finance
#             docs = documents_finance
#             doc_ids = doc_ids_finance
#         elif category == 'insurance':
#             corpus_dict = corpus_dict_insurance
#             docs = documents_insurance
#             doc_ids = doc_ids_insurance
#         elif category == 'faq':
#             corpus_dict = key_to_source_dict
#             docs = faq_documents
#             doc_ids = faq_doc_ids
#         else:
#             logging.warning(f"Unknown category '{category}' for QID: {qid}")
#             answer_dict['answers'].append({"qid": qid, "retrieve": None})
#             continue

#         # Get relevant documents based on source
#         if category == 'faq':
#             relevant_docs = []
#             relevant_doc_ids = []
#             for key in source:
#                 docs_for_key = corpus_dict.get(key, [])
#                 if not docs_for_key:
#                     logging.warning(f"QID: {qid} - No documents found for faq key: {key}")
#                 relevant_docs.extend(docs_for_key)
#                 relevant_doc_ids.extend([key] * len(docs_for_key))
#         else:
#             relevant_docs = []
#             relevant_doc_ids = []
#             for doc_id in source:
#                 docs_for_id = corpus_dict.get(doc_id, [])
#                 if not docs_for_id:
#                     logging.warning(f"QID: {qid} - No documents found for doc_id: {doc_id}")
#                 relevant_docs.extend(docs_for_id)
#                 relevant_doc_ids.extend([doc_id] * len(docs_for_id))

#         if not relevant_docs:
#             logging.warning(f"No valid documents found for QID: {qid}")
#             answer_dict['answers'].append({"qid": qid, "retrieve": None})
#             continue

#         # Tokenize the query
#         tokenized_query = jieba.lcut(query)

#         # BM25 retrieval: get scores for all relevant documents
#         # First, get indices of relevant_docs in all_documents
#         relevant_indices = []
#         for doc_id in source:
#             indices = doc_id_to_indices.get(doc_id, [])
#             if not indices:
#                 logging.warning(f"QID: {qid} - doc_id {doc_id} not found in corpus.")
#             relevant_indices.extend(indices)

#         # If no relevant_indices found, skip
#         if not relevant_indices:
#             logging.warning(f"No relevant indices found for QID: {qid}")
#             answer_dict['answers'].append({"qid": qid, "retrieve": None})
#             continue

#         # Get BM25 scores for all documents
#         scores_all = bm25.get_scores(tokenized_query)

#         # Extract scores for relevant_docs
#         scores_relevant = [scores_all[i] for i in relevant_indices]

#         # Dynamic top_k selection based on len(source)
#         if len(source) < 10:
#             top_k = 4
#         else:
#             top_k = 10
#         # Ensure top_k does not exceed number of relevant documents
#         top_k = min(top_k, len(scores_relevant))

#         # Get top_k relative indices within relevant_docs
#         top_k_relative_indices = np.argsort(scores_relevant)[::-1][:top_k]

#         # Map relative indices to actual indices in all_documents
#         top_k_absolute_indices = [relevant_indices[i] for i in top_k_relative_indices]

#         # Retrieve top_k documents and their doc_ids
#         top_k_docs = [all_documents[i] for i in top_k_absolute_indices]
#         top_k_doc_ids = [all_doc_ids[i] for i in top_k_absolute_indices]

#         # Debug: Log the top_k_doc_ids and their BM25 scores
#         logging.info(f"QID: {qid}, Top {top_k} Doc IDs: {top_k_doc_ids}")

#         # Reranker: compute scores for top_k documents
#         try:
#             reranker_scores = compute_reranker_scores(query, top_k_docs, reranker_model, reranker_tokenizer, device)
#             logging.info(f"QID: {qid}, Reranker scores: {reranker_scores[:5]}")  # Log first 5 scores for brevity
#         except Exception as e:
#             logging.error(f"Error during reranking for QID: {qid}: {e}")
#             answer_dict['answers'].append({"qid": qid, "retrieve": None})
#             continue

#         # Pair documents with scores and doc_ids
#         doc_scores = list(zip(top_k_docs, reranker_scores, top_k_doc_ids))

#         # Sort documents by reranker scores in descending order
#         top_n = 1  # Adjust as needed
#         top_n_docs = sorted(doc_scores, key=lambda x: x[1], reverse=True)[:top_n]

#         if not top_n_docs:
#             logging.warning(f"No documents after reranking for QID: {qid}")
#             answer_dict['answers'].append({"qid": qid, "retrieve": None})
#             continue

#         # Select the best document
#         best_doc, best_score, best_doc_id = top_n_docs[0]
#         logging.info(f"QID: {qid}, Selected Document ID: {best_doc_id}, Score: {best_score:.4f}")

#         # Append to answer dictionary
#         answer_dict['answers'].append({"qid": qid, "retrieve": best_doc_id})

#     # Save results
#     try:
#         with open(args.output_path, 'w', encoding='utf8') as f:
#             json.dump(answer_dict, f, ensure_ascii=False, indent=4)
#         logging.info(f"Results saved to {args.output_path}")
#     except Exception as e:
#         logging.error(f"Error writing output file {args.output_path}: {e}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Two-Stage Retrieval System with BM25 and Reranker Models.')
#     parser.add_argument('--question_path', type=str, required=True, help='Path to the questions JSON file.')
#     parser.add_argument('--source_path', type=str, required=True, help='Path to the source data directory.')
#     parser.add_argument('--output_path', type=str, required=True, help='Path to save the output JSON file.')

#     args = parser.parse_args()

#     main(args)

#--------------------------------------------------------------------------------------------------------------------------------
# bm25+bce_reranker
# import os
# import json
# import argparse
# from tqdm import tqdm
# import jieba
# import pdfplumber
# import torch
# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# from rank_bm25 import BM25Okapi
# import numpy as np
# import logging
# import pytesseract
# from PIL import Image

# # pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
# # 設置日志記錄
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Function to read PDF and split into chunks with OCR fallback
# def read_pdf(pdf_loc, page_infos: list = None):
#     try:
#         pdf = pdfplumber.open(pdf_loc)
#     except Exception as e:
#         logging.error(f"Error opening PDF file {pdf_loc}: {e}")
#         return []
    
#     try:
#         pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
#     except IndexError:
#         logging.warning(f"Page range {page_infos} out of bounds for file {pdf_loc}. Extracting all pages.")
#         pages = pdf.pages
    
#     pdf_text = ''
#     for page_number, page in enumerate(pages, start=1):
#         try:
#             text = page.extract_text()
#             if text:
#                 logging.info(f"Extracted {len(text)} characters from page {page_number} of {pdf_loc} using pdfplumber.")
#                 pdf_text += text + "\n\n"
#             else:
#                 # 如果pdfplumber無法提取文本，嘗試使用OCR
#                 logging.info(f"No text found on page {page_number} of {pdf_loc}. Attempting OCR.")
#                 # 將頁面轉換為圖像
#                 image = page.to_image(resolution=300).original
#                 # 使用PIL將圖像轉換為RGB模式
#                 pil_image = image.convert("RGB")
#                 # 使用pytesseract進行OCR
#                 ocr_text = pytesseract.image_to_string(pil_image, lang='chi_tra')  # 使用繁體中文語言包
#                 if ocr_text.strip():
#                     logging.info(f"Extracted {len(ocr_text)} characters from page {page_number} of {pdf_loc} using OCR.")
#                     pdf_text += ocr_text + "\n\n"
#                 else:
#                     logging.warning(f"OCR failed to extract text from page {page_number} of {pdf_loc}.")
#         except Exception as e:
#             logging.error(f"Error processing page {page_number} in {pdf_loc}: {e}")
#     pdf.close()
    
#     splits = pdf_text.split('\n\n')
#     splits = [split.strip() for split in splits if split.strip()]
#     return splits

# # Function to load data from source path
# def load_data(source_path):
#     masked_file_ls = os.listdir(source_path)
#     corpus_dict = {}
#     all_documents = []
#     all_doc_ids = []
#     missing_pdfs = []
#     for file in tqdm(masked_file_ls, desc=f"Loading data from {source_path}"):
#         try:
#             file_id = int(file.replace('.pdf', ''))
#         except ValueError:
#             logging.warning(f"Skipping non-PDF or improperly named file: {file}")
#             continue
#         file_path = os.path.join(source_path, file)
#         splits = read_pdf(file_path)
#         if not splits:
#             logging.warning(f"No content extracted from file: {file_path}")
#             missing_pdfs.append(file)
#         corpus_dict[file_id] = splits
#         all_documents.extend(splits)
#         all_doc_ids.extend([file_id] * len(splits))
#     if missing_pdfs:
#         logging.info(f"Total missing PDFs: {len(missing_pdfs)}")
#         for pdf in missing_pdfs:
#             logging.info(f"Missing PDF: {pdf}")
#     return corpus_dict, all_documents, all_doc_ids

# # Function to tokenize documents for BM25
# def tokenize_corpus(corpus):
#     tokenized_corpus = [jieba.lcut(doc) for doc in corpus]
#     return tokenized_corpus

# # Function to initialize BM25
# def initialize_bm25(tokenized_corpus):
#     bm25 = BM25Okapi(tokenized_corpus)
#     return bm25

# # Function to load Reranker model
# def load_reranker(model_name, device):
#     try:
#         reranker = AutoModelForSequenceClassification.from_pretrained(model_name)
#         tokenizer = AutoTokenizer.from_pretrained(model_name)
#         reranker.to(device)
#         reranker.eval()
#         logging.info(f"Reranker model '{model_name}' loaded successfully.")
#         return reranker, tokenizer
#     except Exception as e:
#         logging.error(f"Error loading Reranker model {model_name}: {e}")
#         return None, None

# # Function to compute reranker scores using new model
# def compute_reranker_scores(query, documents, reranker_model, reranker_tokenizer, device):
#     reranker_model.eval()
#     sentence_pairs = [[query, doc] for doc in documents]
#     scores = []
#     with torch.no_grad():
#         for i in tqdm(range(0, len(sentence_pairs), 32), desc="Computing Reranker Scores"):
#             batch = sentence_pairs[i:i+32]
#             inputs = reranker_tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
#             inputs = {k: v.to(device) for k, v in inputs.items()}
#             outputs = reranker_model(**inputs)
#             logits = outputs.logits.view(-1).float()
#             # 應用 sigmoid 轉換為概率分數
#             probabilities = torch.sigmoid(logits)
#             batch_scores = probabilities.cpu().numpy()
#             scores.extend(batch_scores)
#     return scores

# def main(args):
#     answer_dict = {"answers": []}

#     # Read questions file
#     try:
#         with open(args.question_path, 'r', encoding='utf8') as f:
#             qs_ref = json.load(f)
#         logging.info(f"Loaded questions from {args.question_path}")
#     except Exception as e:
#         logging.error(f"Error reading question file {args.question_path}: {e}")
#         return

#     # Load reference data
#     corpus_dict_finance, documents_finance, doc_ids_finance = load_data(os.path.join(args.source_path, 'finance'))
#     corpus_dict_insurance, documents_insurance, doc_ids_insurance = load_data(os.path.join(args.source_path, 'insurance'))

#     # Load FAQ mapping and split
#     try:
#         with open(os.path.join(args.source_path, 'faq/pid_map_content.json'), 'rb') as f_s:
#             key_to_source_dict = json.load(f_s)  # 讀取參考資料文件
#             key_to_source_dict = {int(key): value for key, value in key_to_source_dict.items()}
#         logging.info("Loaded FAQ mapping.")
#     except Exception as e:
#         logging.error(f"Error reading FAQ mapping file: {e}")
#         key_to_source_dict = {}
#         faq_documents = []
#         faq_doc_ids = []
    
#     # 檢查是否包含所有需要的 FAQ doc_id
#     required_faq_doc_ids = set()
#     for q in qs_ref.get('questions', []):
#         if 101 <= int(q.get('qid', 0)) <= 150 and q.get('category') == 'faq':
#             required_faq_doc_ids.update(q.get('source', []))
    
#     missing_faq_doc_ids = [doc_id for doc_id in required_faq_doc_ids if doc_id not in key_to_source_dict]
#     if missing_faq_doc_ids:
#         logging.warning(f"The following FAQ doc_ids are missing in 'faq/pid_map_content.json': {missing_faq_doc_ids}")
#     else:
#         logging.info("All required FAQ doc_ids are present in 'faq/pid_map_content.json'.")

#     # Prepare FAQ documents
#     faq_documents = []
#     faq_doc_ids = []
#     for key, value in key_to_source_dict.items():
#         for q in value:
#             combined = f"問題：{q['question']} 答案：{' '.join(q['answers'])}"
#             faq_documents.append(combined)
#             faq_doc_ids.append(key)

#     # Aggregate all documents
#     all_documents = documents_finance + documents_insurance + faq_documents
#     all_doc_ids = doc_ids_finance + doc_ids_insurance + faq_doc_ids

#     # Build a mapping from doc_id to list of indices in all_documents
#     doc_id_to_indices = {}
#     for idx, doc_id in enumerate(all_doc_ids):
#         if doc_id not in doc_id_to_indices:
#             doc_id_to_indices[doc_id] = []
#         doc_id_to_indices[doc_id].append(idx)

#     # Tokenize corpus for BM25
#     tokenized_corpus = tokenize_corpus(all_documents)

#     # Initialize BM25
#     bm25 = initialize_bm25(tokenized_corpus)
#     logging.info("BM25 index initialized.")

#     # Initialize Reranker model
#     reranker_model_name = 'maidalun1020/bce-reranker-base_v1'  # 新的 Reranker 模型
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     reranker_model, reranker_tokenizer = load_reranker(reranker_model_name, device)
#     if reranker_model is None:
#         logging.error("Reranker model failed to load. Exiting.")
#         return

#     # Log loaded doc_ids for verification
#     loaded_doc_ids = set(all_doc_ids)
#     logging.info(f"Total loaded doc_ids: {len(loaded_doc_ids)}")

#     # Process each question
#     for q_dict in tqdm(qs_ref.get('questions', []), desc="Processing questions"):
#         qid = q_dict.get('qid')
#         query = q_dict.get('query')
#         category = q_dict.get('category')
#         source = q_dict.get('source')

#         if not all([qid, query, category, source]):
#             logging.warning(f"Skipping incomplete question entry: {q_dict}")
#             answer_dict['answers'].append({"qid": qid, "retrieve": None})
#             continue

#         # Select the appropriate corpus
#         if category == 'finance':
#             corpus_dict = corpus_dict_finance
#             docs = documents_finance
#             doc_ids = doc_ids_finance
#         elif category == 'insurance':
#             corpus_dict = corpus_dict_insurance
#             docs = documents_insurance
#             doc_ids = doc_ids_insurance
#         elif category == 'faq':
#             corpus_dict = key_to_source_dict
#             docs = faq_documents
#             doc_ids = faq_doc_ids
#         else:
#             logging.warning(f"Unknown category '{category}' for QID: {qid}")
#             answer_dict['answers'].append({"qid": qid, "retrieve": None})
#             continue

#         # Get relevant documents based on source
#         if category == 'faq':
#             relevant_docs = []
#             relevant_doc_ids = []
#             for key in source:
#                 docs_for_key = corpus_dict.get(key, [])
#                 if not docs_for_key:
#                     logging.warning(f"QID: {qid} - No documents found for faq key: {key}")
#                 relevant_docs.extend(docs_for_key)
#                 relevant_doc_ids.extend([key] * len(docs_for_key))
#         else:
#             relevant_docs = []
#             relevant_doc_ids = []
#             for doc_id in source:
#                 docs_for_id = corpus_dict.get(doc_id, [])
#                 if not docs_for_id:
#                     logging.warning(f"QID: {qid} - No documents found for doc_id: {doc_id}")
#                 relevant_docs.extend(docs_for_id)
#                 relevant_doc_ids.extend([doc_id] * len(docs_for_id))

#         if not relevant_docs:
#             logging.warning(f"No valid documents found for QID: {qid}")
#             answer_dict['answers'].append({"qid": qid, "retrieve": None})
#             continue

#         # Tokenize the query
#         tokenized_query = jieba.lcut(query)

#         # BM25 retrieval: get scores for all relevant documents
#         # First, get indices of relevant_docs in all_documents
#         relevant_indices = []
#         for doc_id in source:
#             indices = doc_id_to_indices.get(doc_id, [])
#             if not indices:
#                 logging.warning(f"QID: {qid} - doc_id {doc_id} not found in corpus.")
#             relevant_indices.extend(indices)

#         # If no relevant_indices found, skip
#         if not relevant_indices:
#             logging.warning(f"No relevant indices found for QID: {qid}")
#             answer_dict['answers'].append({"qid": qid, "retrieve": None})
#             continue

#         # Get BM25 scores for all documents
#         scores_all = bm25.get_scores(tokenized_query)

#         # Extract scores for relevant_docs
#         scores_relevant = [scores_all[i] for i in relevant_indices]

#         # Dynamic top_k selection based on len(source)
#         if len(source) < 10:
#             top_k = 4
#         else:
#             top_k = 10
#         # Ensure top_k does not exceed number of relevant documents
#         top_k = min(top_k, len(scores_relevant))

#         # Get top_k relative indices within relevant_docs
#         top_k_relative_indices = np.argsort(scores_relevant)[::-1][:top_k]

#         # Map relative indices to actual indices in all_documents
#         top_k_absolute_indices = [relevant_indices[i] for i in top_k_relative_indices]

#         # Retrieve top_k documents and their doc_ids
#         top_k_docs = [all_documents[i] for i in top_k_absolute_indices]
#         top_k_doc_ids = [all_doc_ids[i] for i in top_k_absolute_indices]

#         # Debug: Log the top_k_doc_ids and their BM25 scores
#         logging.info(f"QID: {qid}, Top {top_k} Doc IDs: {top_k_doc_ids}")

#         # Reranker: compute scores for top_k documents
#         try:
#             reranker_scores = compute_reranker_scores(query, top_k_docs, reranker_model, reranker_tokenizer, device)
#             logging.info(f"QID: {qid}, Reranker scores: {reranker_scores[:5]}")  # Log first 5 scores for brevity
#         except Exception as e:
#             logging.error(f"Error during reranking for QID: {qid}: {e}")
#             answer_dict['answers'].append({"qid": qid, "retrieve": None})
#             continue

#         # Pair documents with scores and doc_ids
#         doc_scores = list(zip(top_k_docs, reranker_scores, top_k_doc_ids))

#         # Sort documents by reranker scores in descending order
#         top_n = 1  # 根據需求調整
#         top_n_docs = sorted(doc_scores, key=lambda x: x[1], reverse=True)[:top_n]

#         if not top_n_docs:
#             logging.warning(f"No documents after reranking for QID: {qid}")
#             answer_dict['answers'].append({"qid": qid, "retrieve": None})
#             continue

#         # Select the best document
#         best_doc, best_score, best_doc_id = top_n_docs[0]
#         logging.info(f"QID: {qid}, Selected Document ID: {best_doc_id}, Score: {best_score:.4f}")

#         # Append to answer dictionary
#         answer_dict['answers'].append({"qid": qid, "retrieve": best_doc_id})

#     # Save results
#     try:
#         with open(args.output_path, 'w', encoding='utf8') as f:
#             json.dump(answer_dict, f, ensure_ascii=False, indent=4)
#         logging.info(f"Results saved to {args.output_path}")
#     except Exception as e:
#         logging.error(f"Error writing output file {args.output_path}: {e}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Two-Stage Retrieval System with BM25 and Reranker Models.')
#     parser.add_argument('--question_path', type=str, required=True, help='Path to the questions JSON file.')
#     parser.add_argument('--source_path', type=str, required=True, help='Path to the source data directory.')
#     parser.add_argument('--output_path', type=str, required=True, help='Path to save the output JSON file.')

#     args = parser.parse_args()

#     main(args)

#--------------------------------------------------------------------------------------------------------------------------------
# import os
# import json
# import argparse
# from tqdm import tqdm
# import jieba
# import pdfplumber
# import torch
# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# from rank_bm25 import BM25Okapi
# import numpy as np
# import logging
# import pytesseract
# from PIL import Image
# from sentence_transformers import SentenceTransformer

# pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
# # 設置日志記錄
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Function to read PDF and split into chunks with OCR fallback
# def read_pdf(pdf_loc, page_infos: list = None):
#     try:
#         pdf = pdfplumber.open(pdf_loc)
#     except Exception as e:
#         logging.error(f"Error opening PDF file {pdf_loc}: {e}")
#         return []
    
#     try:
#         pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
#     except IndexError:
#         logging.warning(f"Page range {page_infos} out of bounds for file {pdf_loc}. Extracting all pages.")
#         pages = pdf.pages
    
#     pdf_text = ''
#     for page_number, page in enumerate(pages, start=1):
#         try:
#             text = page.extract_text()
#             if text:
#                 logging.info(f"Extracted {len(text)} characters from page {page_number} of {pdf_loc} using pdfplumber.")
#                 pdf_text += text + "\n\n"
#             else:
#                 # 如果pdfplumber無法提取文本，嘗試使用OCR
#                 logging.info(f"No text found on page {page_number} of {pdf_loc}. Attempting OCR.")
#                 # 將頁面轉換為圖像
#                 image = page.to_image(resolution=300).original
#                 # 使用PIL將圖像轉換為RGB模式
#                 pil_image = image.convert("RGB")
#                 # 使用pytesseract進行OCR
#                 ocr_text = pytesseract.image_to_string(pil_image, lang='chi_tra')  # 使用繁體中文語言包
#                 if ocr_text.strip():
#                     logging.info(f"Extracted {len(ocr_text)} characters from page {page_number} of {pdf_loc} using OCR.")
#                     pdf_text += ocr_text + "\n\n"
#                 else:
#                     logging.warning(f"OCR failed to extract text from page {page_number} of {pdf_loc}.")
#         except Exception as e:
#             logging.error(f"Error processing page {page_number} in {pdf_loc}: {e}")
#     pdf.close()
    
#     splits = pdf_text.split('\n\n')
#     splits = [split.strip() for split in splits if split.strip()]
#     return splits

# # Function to load data from source path
# def load_data(source_path):
#     masked_file_ls = os.listdir(source_path)
#     corpus_dict = {}
#     all_documents = []
#     all_doc_ids = []
#     missing_pdfs = []
#     for file in tqdm(masked_file_ls, desc=f"Loading data from {source_path}"):
#         try:
#             file_id = int(file.replace('.pdf', ''))
#         except ValueError:
#             logging.warning(f"Skipping non-PDF or improperly named file: {file}")
#             continue
#         file_path = os.path.join(source_path, file)
#         splits = read_pdf(file_path)
#         if not splits:
#             logging.warning(f"No content extracted from file: {file_path}")
#             missing_pdfs.append(file)
#         corpus_dict[file_id] = splits
#         all_documents.extend(splits)
#         all_doc_ids.extend([file_id] * len(splits))
#     if missing_pdfs:
#         logging.info(f"Total missing PDFs: {len(missing_pdfs)}")
#         for pdf in missing_pdfs:
#             logging.info(f"Missing PDF: {pdf}")
#     return corpus_dict, all_documents, all_doc_ids

# # Function to tokenize documents for BM25
# def tokenize_corpus(corpus):
#     tokenized_corpus = [jieba.lcut(doc) for doc in corpus]
#     return tokenized_corpus

# # Function to initialize BM25
# def initialize_bm25(tokenized_corpus):
#     bm25 = BM25Okapi(tokenized_corpus)
#     return bm25

# # Function to load Reranker1 model (SentenceTransformer-based)
# def load_reranker1(device):
#     try:
#         model = SentenceTransformer('BAAI/bge-large-zh-v1.5', device=device)
#         logging.info("Reranker1 model (SentenceTransformer) loaded successfully.")
#         return model
#     except Exception as e:
#         logging.error(f"Error loading Reranker1 model: {e}")
#         return None

# # Function to load Reranker2 model
# def load_reranker2(model_name, device):
#     try:
#         reranker = AutoModelForSequenceClassification.from_pretrained(model_name)
#         tokenizer = AutoTokenizer.from_pretrained(model_name)
#         reranker.to(device)
#         reranker.eval()
#         logging.info(f"Reranker2 model '{model_name}' loaded successfully.")
#         return reranker, tokenizer
#     except Exception as e:
#         logging.error(f"Error loading Reranker2 model {model_name}: {e}")
#         return None, None

# # Function to compute reranker2 scores using AutoModelForSequenceClassification
# def compute_reranker2_scores(query, documents, reranker_model, reranker_tokenizer, device):
#     reranker_model.eval()
#     sentence_pairs = [[query, doc] for doc in documents]
#     scores = []
#     with torch.no_grad():
#         for i in tqdm(range(0, len(sentence_pairs), 32), desc="Computing Reranker2 Scores"):
#             batch = sentence_pairs[i:i+32]
#             inputs = reranker_tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
#             inputs = {k: v.to(device) for k, v in inputs.items()}
#             outputs = reranker_model(**inputs)
#             logits = outputs.logits.view(-1).float()
#             # 應用 sigmoid 轉換為概率分數
#             probabilities = torch.sigmoid(logits)
#             batch_scores = probabilities.cpu().numpy()
#             scores.extend(batch_scores)
#     return scores

# def main(args):
#     answer_dict = {"answers": []}

#     # Read questions file
#     try:
#         with open(args.question_path, 'r', encoding='utf8') as f:
#             qs_ref = json.load(f)
#         logging.info(f"Loaded questions from {args.question_path}")
#     except Exception as e:
#         logging.error(f"Error reading question file {args.question_path}: {e}")
#         return

#     # Load reference data
#     corpus_dict_finance, documents_finance, doc_ids_finance = load_data(os.path.join(args.source_path, 'finance'))
#     corpus_dict_insurance, documents_insurance, doc_ids_insurance = load_data(os.path.join(args.source_path, 'insurance'))

#     # Load FAQ mapping and split
#     try:
#         with open(os.path.join(args.source_path, 'faq/pid_map_content.json'), 'rb') as f_s:
#             key_to_source_dict = json.load(f_s)  # 讀取參考資料文件
#             key_to_source_dict = {int(key): value for key, value in key_to_source_dict.items()}
#         logging.info("Loaded FAQ mapping.")
#     except Exception as e:
#         logging.error(f"Error reading FAQ mapping file: {e}")
#         key_to_source_dict = {}
#         faq_documents = []
#         faq_doc_ids = []
    
#     # 檢查是否包含所有需要的 FAQ doc_id
#     required_faq_doc_ids = set()
#     for q in qs_ref.get('questions', []):
#         if 101 <= int(q.get('qid', 0)) <= 150 and q.get('category') == 'faq':
#             required_faq_doc_ids.update(q.get('source', []))
    
#     missing_faq_doc_ids = [doc_id for doc_id in required_faq_doc_ids if doc_id not in key_to_source_dict]
#     if missing_faq_doc_ids:
#         logging.warning(f"The following FAQ doc_ids are missing in 'faq/pid_map_content.json': {missing_faq_doc_ids}")
#     else:
#         logging.info("All required FAQ doc_ids are present in 'faq/pid_map_content.json'.")

#     # Prepare FAQ documents
#     faq_documents = []
#     faq_doc_ids = []
#     for key, value in key_to_source_dict.items():
#         for q in value:
#             combined = f"問題：{q['question']} 答案：{' '.join(q['answers'])}"
#             faq_documents.append(combined)
#             faq_doc_ids.append(key)

#     # Aggregate all documents
#     all_documents = documents_finance + documents_insurance + faq_documents
#     all_doc_ids = doc_ids_finance + doc_ids_insurance + faq_doc_ids

#     # Build a mapping from doc_id to list of indices in all_documents
#     doc_id_to_indices = {}
#     for idx, doc_id in enumerate(all_doc_ids):
#         if doc_id not in doc_id_to_indices:
#             doc_id_to_indices[doc_id] = []
#         doc_id_to_indices[doc_id].append(idx)

#     # Tokenize corpus for BM25
#     tokenized_corpus = tokenize_corpus(all_documents)

#     # Initialize BM25
#     bm25 = initialize_bm25(tokenized_corpus)
#     logging.info("BM25 index initialized.")

#     # Initialize Reranker models
#     device =  'cpu'

#     # Initialize Reranker1 (SentenceTransformer)
#     reranker1_model = load_reranker1(device)
#     if reranker1_model is None:
#         logging.error("Reranker1 model failed to load. Exiting.")
#         return

#     # Initialize Reranker2
#     reranker2_model_name = 'maidalun1020/bce-reranker-base_v1'  # Existing Reranker model
#     reranker2_model, reranker2_tokenizer = load_reranker2(reranker2_model_name, device)
#     if reranker2_model is None:
#         logging.error("Reranker2 model failed to load. Exiting.")
#         return

#     # Log loaded doc_ids for verification
#     loaded_doc_ids = set(all_doc_ids)
#     logging.info(f"Total loaded doc_ids: {len(loaded_doc_ids)}")

#     # Process each question
#     for q_dict in tqdm(qs_ref.get('questions', []), desc="Processing questions"):
#         qid = q_dict.get('qid')
#         query = q_dict.get('query')
#         category = q_dict.get('category')
#         source = q_dict.get('source')

#         if not all([qid, query, category, source]):
#             logging.warning(f"Skipping incomplete question entry: {q_dict}")
#             answer_dict['answers'].append({"qid": qid, "retrieve": None})
#             continue

#         # Select the appropriate corpus
#         if category == 'finance':
#             corpus_dict = corpus_dict_finance
#             docs = documents_finance
#             doc_ids = doc_ids_finance
#         elif category == 'insurance':
#             corpus_dict = corpus_dict_insurance
#             docs = documents_insurance
#             doc_ids = doc_ids_insurance
#         elif category == 'faq':
#             corpus_dict = key_to_source_dict
#             docs = faq_documents
#             doc_ids = faq_doc_ids
#         else:
#             logging.warning(f"Unknown category '{category}' for QID: {qid}")
#             answer_dict['answers'].append({"qid": qid, "retrieve": None})
#             continue

#         # Get relevant documents based on source
#         if category == 'faq':
#             relevant_docs = []
#             relevant_doc_ids = []
#             for key in source:
#                 docs_for_key = corpus_dict.get(key, [])
#                 if not docs_for_key:
#                     logging.warning(f"QID: {qid} - No documents found for faq key: {key}")
#                 relevant_docs.extend(docs_for_key)
#                 relevant_doc_ids.extend([key] * len(docs_for_key))
#         else:
#             relevant_docs = []
#             relevant_doc_ids = []
#             for doc_id in source:
#                 docs_for_id = corpus_dict.get(doc_id, [])
#                 if not docs_for_id:
#                     logging.warning(f"QID: {qid} - No documents found for doc_id: {doc_id}")
#                 relevant_docs.extend(docs_for_id)
#                 relevant_doc_ids.extend([doc_id] * len(docs_for_id))

#         if not relevant_docs:
#             logging.warning(f"No valid documents found for QID: {qid}")
#             answer_dict['answers'].append({"qid": qid, "retrieve": None})
#             continue

#         # Tokenize the query for BM25
#         tokenized_query = jieba.lcut(query)

#         # BM25 retrieval: get scores for all relevant documents
#         # First, get indices of relevant_docs in all_documents
#         relevant_indices = []
#         for doc_id in source:
#             indices = doc_id_to_indices.get(doc_id, [])
#             if not indices:
#                 logging.warning(f"QID: {qid} - doc_id {doc_id} not found in corpus.")
#             relevant_indices.extend(indices)

#         # If no relevant_indices found, skip
#         if not relevant_indices:
#             logging.warning(f"No relevant indices found for QID: {qid}")
#             answer_dict['answers'].append({"qid": qid, "retrieve": None})
#             continue

#         # Get BM25 scores for all documents
#         scores_all = bm25.get_scores(tokenized_query)

#         # Extract scores for relevant_docs
#         scores_relevant = [scores_all[i] for i in relevant_indices]

#         # Dynamic top_k selection based on len(source)
#         if len(source) < 10:
#             top_k = 4
#         else:
#             top_k = 10
#         # Ensure top_k does not exceed number of relevant documents
#         top_k = min(top_k, len(scores_relevant))

#         # Get top_k relative indices within relevant_docs
#         top_k_relative_indices = np.argsort(scores_relevant)[::-1][:top_k]

#         # Map relative indices to actual indices in all_documents
#         top_k_absolute_indices = [relevant_indices[i] for i in top_k_relative_indices]

#         # Retrieve top_k documents and their doc_ids
#         top_k_docs = [all_documents[i] for i in top_k_absolute_indices]
#         top_k_doc_ids = [all_doc_ids[i] for i in top_k_absolute_indices]

#         # Debug: Log the top_k_doc_ids and their BM25 scores
#         logging.info(f"QID: {qid}, Top {top_k} Doc IDs: {top_k_doc_ids}")

#         # Reranker1: compute scores using SentenceTransformer
#         try:
#             instruction = "为这个句子生成表示以用于检索相关文章："
#             augmented_queries = [instruction + query]  # Batch size 1
#             # Encode query and passages
#             q_embeddings = reranker1_model.encode(augmented_queries, normalize_embeddings=True)
#             p_embeddings = reranker1_model.encode(top_k_docs, normalize_embeddings=True)
#             # Compute cosine similarity (dot product since embeddings are normalized)
#             reranker1_scores = (q_embeddings @ p_embeddings.T).flatten().tolist()
#             logging.info(f"QID: {qid}, Reranker1 scores: {reranker1_scores[:5]}")  # Log first 5 scores for brevity
#         except Exception as e:
#             logging.error(f"Error during reranking with Reranker1 for QID: {qid}: {e}")
#             answer_dict['answers'].append({"qid": qid, "retrieve": None})
#             continue

#         # Reranker2: compute scores using AutoModelForSequenceClassification
#         try:
#             reranker2_scores = compute_reranker2_scores(query, top_k_docs, reranker2_model, reranker2_tokenizer, device)
#             logging.info(f"QID: {qid}, Reranker2 scores: {reranker2_scores[:5]}")  # Log first 5 scores for brevity
#         except Exception as e:
#             logging.error(f"Error during reranking with Reranker2 for QID: {qid}: {e}")
#             answer_dict['answers'].append({"qid": qid, "retrieve": None})
#             continue

#         # Average the scores
#         if len(reranker1_scores) != len(top_k_docs) or len(reranker2_scores) != len(top_k_docs):
#             logging.error(f"Score length mismatch for QID: {qid}")
#             answer_dict['answers'].append({"qid": qid, "retrieve": None})
#             continue

#         average_scores = [(r1 + r2) / 2 for r1, r2 in zip(reranker1_scores, reranker2_scores)]

#         # Pair documents with average scores and doc_ids
#         doc_scores = list(zip(top_k_docs, average_scores, top_k_doc_ids))

#         # Sort documents by average scores in descending order
#         top_n = 1  # 根據需求調整
#         top_n_docs = sorted(doc_scores, key=lambda x: x[1], reverse=True)[:top_n]

#         if not top_n_docs:
#             logging.warning(f"No documents after reranking for QID: {qid}")
#             answer_dict['answers'].append({"qid": qid, "retrieve": None})
#             continue

#         # Select the best document
#         best_doc, best_score, best_doc_id = top_n_docs[0]
#         logging.info(f"QID: {qid}, Selected Document ID: {best_doc_id}, Average Score: {best_score:.4f}")

#         # Append to answer dictionary
#         answer_dict['answers'].append({"qid": qid, "retrieve": best_doc_id})

#     # Save results
#     try:
#         with open(args.output_path, 'w', encoding='utf8') as f:
#             json.dump(answer_dict, f, ensure_ascii=False, indent=4)
#         logging.info(f"Results saved to {args.output_path}")
#     except Exception as e:
#         logging.error(f"Error writing output file {args.output_path}: {e}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Two-Reranker Retrieval System with SentenceTransformer and BCE Reranker.')
#     parser.add_argument('--question_path', type=str, required=True, help='Path to the questions JSON file.')
#     parser.add_argument('--source_path', type=str, required=True, help='Path to the source data directory.')
#     parser.add_argument('--output_path', type=str, required=True, help='Path to save the output JSON file.')

#     args = parser.parse_args()

#     main(args)

#--------------------------------------------------------------------------------------------------------------------------------

import os
import json
import argparse
from tqdm import tqdm
import jieba
import pdfplumber
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from rank_bm25 import BM25Okapi
import numpy as np
import logging
import pytesseract
from PIL import Image
from sentence_transformers import SentenceTransformer
import nltk
from pkuseg import pkuseg
import faiss
from collections import defaultdict

# Download NLTK data
nltk.download('punkt')

# Initialize pkuseg for Chinese sentence segmentation
seg = pkuseg()

# pytesseract path (ensure this path is correct on your system)
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to read PDF and split into coherent chunks with OCR fallback
def read_pdf(pdf_loc, page_infos: list = None, max_tokens=512, overlap_tokens=50):
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
                logging.info(f"No text found on page {page_number} of {pdf_loc}. Attempting OCR.")
                image = page.to_image(resolution=300).original
                pil_image = image.convert("RGB")
                ocr_text = pytesseract.image_to_string(pil_image, lang='chi_tra')
                if ocr_text.strip():
                    logging.info(f"Extracted {len(ocr_text)} characters from page {page_number} of {pdf_loc} using OCR.")
                    pdf_text += ocr_text + "\n\n"
                else:
                    logging.warning(f"OCR failed to extract text from page {page_number} of {pdf_loc}.")
        except Exception as e:
            logging.error(f"Error processing page {page_number} in {pdf_loc}: {e}")
    pdf.close()
    
    # Semantic chunking using pkuseg for sentence segmentation
    sentences = seg.cut(pdf_text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        tokens = jieba.lcut(sentence)
        token_length = len(tokens)
        
        if current_length + token_length > max_tokens:
            if current_chunk:
                chunk = ' '.join(current_chunk)
                chunks.append(chunk)
                # Implement overlapping
                if overlap_tokens > 0:
                    current_chunk = current_chunk[-overlap_tokens:]
                    current_length = sum(len(jieba.lcut(s)) for s in current_chunk)
                else:
                    current_chunk = []
                    current_length = 0
        current_chunk.append(sentence)
        current_length += token_length
    
    # Add any remaining sentences as the last chunk
    if current_chunk:
        chunk = ' '.join(current_chunk)
        chunks.append(chunk)
    
    logging.info(f"Total chunks created from {pdf_loc}: {len(chunks)}")
    return chunks

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

# Function to tokenize documents for BM25
def tokenize_corpus(corpus):
    tokenized_corpus = [jieba.lcut(doc) for doc in corpus]
    return tokenized_corpus

# Function to initialize BM25 with customizable parameters
def initialize_bm25(tokenized_corpus, k1=1.5, b=0.75):
    bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)
    return bm25

# Function to load Reranker1 model (SentenceTransformer-based)
def load_reranker1(device):
    try:
        model = SentenceTransformer('BAAI/bge-large-zh-v1.5', device=device)
        logging.info("Reranker1 model (SentenceTransformer) loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading Reranker1 model: {e}")
        return None

# Function to load Reranker2 model
def load_reranker2(model_name, device):
    try:
        reranker = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        reranker.to(device)
        reranker.eval()
        logging.info(f"Reranker2 model '{model_name}' loaded successfully.")
        return reranker, tokenizer
    except Exception as e:
        logging.error(f"Error loading Reranker2 model {model_name}: {e}")
        return None, None

# Function to compute reranker2 scores using AutoModelForSequenceClassification
def compute_reranker2_scores(query, documents, reranker_model, reranker_tokenizer, device, batch_size=32):
    reranker_model.eval()
    sentence_pairs = [[query, doc] for doc in documents]
    scores = []
    with torch.no_grad():
        for i in tqdm(range(0, len(sentence_pairs), batch_size), desc="Computing Reranker2 Scores"):
            batch = sentence_pairs[i:i+batch_size]
            inputs = reranker_tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = reranker_model(**inputs)
            logits = outputs.logits.view(-1).float()
            # Apply sigmoid to get probability scores
            probabilities = torch.sigmoid(logits)
            batch_scores = probabilities.cpu().numpy()
            scores.extend(batch_scores)
    return scores

# Function to build FAISS index
def build_faiss_index(embeddings, dimension, index_path=None):
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)  # Ensure embeddings are normalized for cosine similarity
    index.add(embeddings)
    if index_path:
        faiss.write_index(index, index_path)
        logging.info(f"FAISS index saved to {index_path}")
    return index

# Function to load FAISS index
def load_faiss_index(index_path):
    index = faiss.read_index(index_path)
    logging.info(f"FAISS index loaded from {index_path}")
    return index

# Function to query FAISS
def query_faiss(index, query_embedding, top_k=10):
    faiss.normalize_L2(query_embedding)
    D, I = index.search(query_embedding, top_k)
    return I[0], D[0]

# Function to aggregate scores for documents with multiple chunks
def aggregate_scores(doc_ids, chunk_scores, method='max'):
    doc_score_dict = defaultdict(list)
    for doc_id, score in zip(doc_ids, chunk_scores):
        doc_score_dict[doc_id].append(score)
    
    if method == 'max':
        aggregated_scores = {doc_id: max(scores) for doc_id, scores in doc_score_dict.items()}
    elif method == 'mean':
        aggregated_scores = {doc_id: np.mean(scores) for doc_id, scores in doc_score_dict.items()}
    elif method == 'sum':
        aggregated_scores = {doc_id: np.sum(scores) for doc_id, scores in doc_score_dict.items()}
    else:
        raise ValueError("Invalid aggregation method. Choose from 'max', 'mean', 'sum'.")
    
    return aggregated_scores

def main(args):
    answer_dict = {"answers": []}

    # Read questions file
    try:
        with open(args.question_path, 'r', encoding='utf8') as f:
            qs_ref = json.load(f)
        logging.info(f"Loaded questions from {args.question_path}")
    except Exception as e:
        logging.error(f"Error reading question file {args.question_path}: {e}")
        return

    # Load reference data
    corpus_dict_finance, documents_finance, doc_ids_finance = load_data(os.path.join(args.source_path, 'finance'))
    corpus_dict_insurance, documents_insurance, doc_ids_insurance = load_data(os.path.join(args.source_path, 'insurance'))

    # Load FAQ mapping and split
    try:
        with open(os.path.join(args.source_path, 'faq/pid_map_content.json'), 'rb') as f_s:
            key_to_source_dict = json.load(f_s)  # Read reference data file
            key_to_source_dict = {int(key): value for key, value in key_to_source_dict.items()}
        logging.info("Loaded FAQ mapping.")
    except Exception as e:
        logging.error(f"Error reading FAQ mapping file: {e}")
        key_to_source_dict = {}
        faq_documents = []
        faq_doc_ids = []
    
    # Check for required FAQ doc_ids
    required_faq_doc_ids = set()
    for q in qs_ref.get('questions', []):
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

    # Build a mapping from doc_id to list of indices in all_documents
    doc_id_to_indices = {}
    for idx, doc_id in enumerate(all_doc_ids):
        if doc_id not in doc_id_to_indices:
            doc_id_to_indices[doc_id] = []
        doc_id_to_indices[doc_id].append(idx)

    # Tokenize corpus for BM25
    tokenized_corpus = tokenize_corpus(all_documents)

    # Initialize BM25 with customizable parameters
    bm25 = initialize_bm25(tokenized_corpus, k1=args.bm25_k1, b=args.bm25_b)
    logging.info("BM25 index initialized.")

    # Initialize Reranker models
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")

    # Initialize Reranker1 (SentenceTransformer)
    reranker1_model = load_reranker1(device)
    if reranker1_model is None:
        logging.error("Reranker1 model failed to load. Exiting.")
        return

    # Initialize Reranker2
    reranker2_model_name = 'maidalun1020/bce-reranker-base_v1'  # Existing Reranker model
    reranker2_model, reranker2_tokenizer = load_reranker2(reranker2_model_name, device)
    if reranker2_model is None:
        logging.error("Reranker2 model failed to load. Exiting.")
        return

    # Log loaded doc_ids for verification
    loaded_doc_ids = set(all_doc_ids)
    logging.info(f"Total loaded doc_ids: {len(loaded_doc_ids)}")

    # Optionally, build FAISS index for SentenceTransformer embeddings
    # Uncomment if you want to use FAISS for initial dense retrieval
    # embedding_model = reranker1_model
    # embeddings = embedding_model.encode(all_documents, normalize_embeddings=True, show_progress_bar=True)
    # faiss_index = build_faiss_index(embeddings, embeddings.shape[1], index_path=args.faiss_index_path)

    # Process each question
    for q_dict in tqdm(qs_ref.get('questions', []), desc="Processing questions"):
        qid = q_dict.get('qid')
        query = q_dict.get('query')
        category = q_dict.get('category')
        source = q_dict.get('source')

        if not all([qid, query, category, source]):
            logging.warning(f"Skipping incomplete question entry: {q_dict}")
            answer_dict['answers'].append({"qid": qid, "retrieve": None})
            continue

        # Select the appropriate corpus
        if category == 'finance':
            corpus_dict = corpus_dict_finance
            docs = documents_finance
            doc_ids = doc_ids_finance
        elif category == 'insurance':
            corpus_dict = corpus_dict_insurance
            docs = documents_insurance
            doc_ids = doc_ids_insurance
        elif category == 'faq':
            corpus_dict = key_to_source_dict
            docs = faq_documents
            doc_ids = faq_doc_ids
        else:
            logging.warning(f"Unknown category '{category}' for QID: {qid}")
            answer_dict['answers'].append({"qid": qid, "retrieve": None})
            continue

        # Get relevant documents based on source
        if category == 'faq':
            relevant_docs = []
            relevant_doc_ids = []
            for key in source:
                docs_for_key = corpus_dict.get(key, [])
                if not docs_for_key:
                    logging.warning(f"QID: {qid} - No documents found for faq key: {key}")
                relevant_docs.extend(docs_for_key)
                relevant_doc_ids.extend([key] * len(docs_for_key))
        else:
            relevant_docs = []
            relevant_doc_ids = []
            for doc_id in source:
                docs_for_id = corpus_dict.get(doc_id, [])
                if not docs_for_id:
                    logging.warning(f"QID: {qid} - No documents found for doc_id: {doc_id}")
                relevant_docs.extend(docs_for_id)
                relevant_doc_ids.extend([doc_id] * len(docs_for_id))

        if not relevant_docs:
            logging.warning(f"No valid documents found for QID: {qid}")
            answer_dict['answers'].append({"qid": qid, "retrieve": None})
            continue

        # Tokenize the query for BM25
        tokenized_query = jieba.lcut(query)

        # BM25 retrieval: get scores for all documents
        scores_all = bm25.get_scores(tokenized_query)

        # Get indices of relevant_docs in all_documents
        relevant_indices = []
        for doc_id in source:
            indices = doc_id_to_indices.get(doc_id, [])
            if not indices:
                logging.warning(f"QID: {qid} - doc_id {doc_id} not found in corpus.")
            relevant_indices.extend(indices)

        # Extract scores for relevant_docs
        scores_relevant = [scores_all[i] for i in relevant_indices]

        # Dynamic top_k selection based on len(source)
        if len(source) < 10:
            top_k = 4
        else:
            top_k = 10
        # Ensure top_k does not exceed number of relevant documents
        top_k = min(top_k, len(scores_relevant))

        # Get top_k relative indices within relevant_docs
        top_k_relative_indices = np.argsort(scores_relevant)[::-1][:top_k]

        # Map relative indices to actual indices in all_documents
        top_k_absolute_indices = [relevant_indices[i] for i in top_k_relative_indices]

        # Retrieve top_k documents and their doc_ids
        top_k_docs = [all_documents[i] for i in top_k_absolute_indices]
        top_k_doc_ids = [all_doc_ids[i] for i in top_k_absolute_indices]

        # Debug: Log the top_k_doc_ids and their BM25 scores
        logging.info(f"QID: {qid}, Top {top_k} Doc IDs: {top_k_doc_ids}")

        # Reranker1: compute scores using SentenceTransformer
        try:
            instruction = "为这个句子生成表示以用于检索相关文章："
            augmented_queries = [instruction + query]  # Batch size 1
            # Encode query and passages
            q_embeddings = reranker1_model.encode(augmented_queries, normalize_embeddings=True)
            p_embeddings = reranker1_model.encode(top_k_docs, normalize_embeddings=True)
            # Compute cosine similarity (dot product since embeddings are normalized)
            reranker1_scores = (q_embeddings @ p_embeddings.T).flatten().tolist()
            logging.info(f"QID: {qid}, Reranker1 scores: {reranker1_scores[:5]}")  # Log first 5 scores for brevity
        except Exception as e:
            logging.error(f"Error during reranking with Reranker1 for QID: {qid}: {e}")
            answer_dict['answers'].append({"qid": qid, "retrieve": None})
            continue

        # Reranker2: compute scores using AutoModelForSequenceClassification
        try:
            reranker2_scores = compute_reranker2_scores(query, top_k_docs, reranker2_model, reranker2_tokenizer, device)
            logging.info(f"QID: {qid}, Reranker2 scores: {reranker2_scores[:5]}")  # Log first 5 scores for brevity
        except Exception as e:
            logging.error(f"Error during reranking with Reranker2 for QID: {qid}: {e}")
            answer_dict['answers'].append({"qid": qid, "retrieve": None})
            continue

        # Aggregate scores per document for Reranker1 and Reranker2
        aggregated_reranker1 = aggregate_scores(top_k_doc_ids, reranker1_scores, method='max')
        aggregated_reranker2 = aggregate_scores(top_k_doc_ids, reranker2_scores, method='max')

        # Define weights for combining reranker scores
        weight_reranker1 = args.reranker1_weight
        weight_reranker2 = args.reranker2_weight

        # Combine aggregated scores with weights
        final_scores = {}
        for doc_id in aggregated_reranker1:
            if doc_id in aggregated_reranker2:
                final_scores[doc_id] = (weight_reranker1 * aggregated_reranker1[doc_id] +
                                         weight_reranker2 * aggregated_reranker2[doc_id])
            else:
                final_scores[doc_id] = weight_reranker1 * aggregated_reranker1[doc_id]

        # Sort documents by final_scores in descending order
        sorted_docs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

        # Select the top document
        if sorted_docs:
            best_doc_id, best_score = sorted_docs[0]
            logging.info(f"QID: {qid}, Selected Document ID: {best_doc_id}, Final Score: {best_score:.4f}")
            answer_dict['answers'].append({"qid": qid, "retrieve": best_doc_id})
        else:
            logging.warning(f"No documents after reranking for QID: {qid}")
            answer_dict['answers'].append({"qid": qid, "retrieve": None})

    # Save results
    try:
        with open(args.output_path, 'w', encoding='utf8') as f:
            json.dump(answer_dict, f, ensure_ascii=False, indent=4)
        logging.info(f"Results saved to {args.output_path}")
    except Exception as e:
        logging.error(f"Error writing output file {args.output_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enhanced Retrieval System for Long Documents.')
    parser.add_argument('--question_path', type=str, required=True, help='Path to the questions JSON file.')
    parser.add_argument('--source_path', type=str, required=True, help='Path to the source data directory.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output JSON file.')
    parser.add_argument('--bm25_k1', type=float, default=1.5, help='BM25 k1 parameter.')
    parser.add_argument('--bm25_b', type=float, default=0.75, help='BM25 b parameter.')
    parser.add_argument('--max_sentences', type=int, default=10, help='Maximum number of sentences per chunk.')
    parser.add_argument('--overlap', type=int, default=2, help='Number of overlapping sentences between chunks.')
    parser.add_argument('--reranker1_weight', type=float, default=0.5, help='Weight for Reranker1 scores.')
    parser.add_argument('--reranker2_weight', type=float, default=0.5, help='Weight for Reranker2 scores.')
    # Add FAISS index path if using FAISS-based retrieval
    # parser.add_argument('--faiss_index_path', type=str, default=None, help='Path to save/load FAISS index.')

    args = parser.parse_args()

    main(args)
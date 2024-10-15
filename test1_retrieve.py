# import os
# import json
# import argparse
# import logging

# from tqdm import tqdm
# import jieba  # 用於中文文本分詞
# import pdfplumber  # 用於從PDF文件中提取文字的工具
# from rank_bm25 import BM25Okapi  # 使用BM25演算法進行文件檢索

# # 配置日誌設置
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

# # 創建控制台處理器
# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.INFO)

# # 創建文件處理器
# file_handler = logging.FileHandler('pdf_extraction.log', mode='w', encoding='utf-8')
# file_handler.setLevel(logging.DEBUG)

# # 創建日誌格式
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# console_handler.setFormatter(formatter)
# file_handler.setFormatter(formatter)

# # 添加處理器到日誌器
# logger.addHandler(console_handler)
# logger.addHandler(file_handler)


# # 載入參考資料，返回一個字典，key為檔案名稱，value為PDF檔內容的文本
# def load_data(source_path):
#     logger.info(f"開始載入資料夾: {source_path}")
#     try:
#         masked_file_ls = os.listdir(source_path)  # 獲取資料夾中的檔案列表
#     except Exception as e:
#         logger.error(f"無法列出目錄 {source_path}: {e}")
#         raise

#     corpus_dict = {}
#     for file in tqdm(masked_file_ls, desc="載入PDF文件"):
#         if not file.lower().endswith('.pdf'):
#             logger.warning(f"跳過非PDF文件: {file}")
#             continue
#         file_id = int(file.replace('.pdf', ''))
#         file_path = os.path.join(source_path, file)
#         logger.debug(f"處理文件: {file_path}")
#         try:
#             text = read_pdf(file_path)
#             corpus_dict[file_id] = text
#             logger.info(f"成功提取文件 {file} 的文本內容")
#         except Exception as e:
#             logger.error(f"提取文件 {file} 時出錯: {e}")
#     logger.info(f"完成載入資料夾: {source_path}")
#     return corpus_dict


# # 讀取單個PDF文件並返回其文本內容
# def read_pdf(pdf_loc, page_infos: list = None):
#     logger.debug(f"打開PDF文件: {pdf_loc}")
#     try:
#         pdf = pdfplumber.open(pdf_loc)  # 打開指定的PDF文件
#     except Exception as e:
#         logger.error(f"無法打開PDF文件 {pdf_loc}: {e}")
#         raise

#     try:
#         # 如果指定了頁面範圍，則只提取該範圍的頁面，否則提取所有頁面
#         pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
#         pdf_text = ''
#         for idx, page in enumerate(pages):
#             text = page.extract_text()  # 提取頁面的文本內容
#             if text:
#                 pdf_text += text
#             else:
#                 logger.warning(f"頁面 {idx + 1} 無法提取文本")
#         logger.debug(f"完成提取PDF文件: {pdf_loc}")
#         return pdf_text  # 返回萃取出的文本
#     except Exception as e:
#         logger.error(f"提取PDF文件 {pdf_loc} 時出錯: {e}")
#         raise
#     finally:
#         pdf.close()  # 關閉PDF文件
#         logger.debug(f"已關閉PDF文件: {pdf_loc}")


# # 根據查詢語句和指定的來源，檢索答案
# def BM25_retrieve(qs, source, corpus_dict):
#     logger.debug(f"開始BM25檢索，查詢: {qs}, 來源: {source}")
#     try:
#         filtered_corpus = [corpus_dict[int(file)] for file in source]
#     except KeyError as e:
#         logger.error(f"來源中存在無效的檔案ID: {e}")
#         raise

#     # [TODO] 可自行替換其他檢索方式，以提升效能

#     tokenized_corpus = [list(jieba.cut_for_search(doc)) for doc in filtered_corpus]  # 將每篇文檔進行分詞
#     bm25 = BM25Okapi(tokenized_corpus)  # 使用BM25演算法建立檢索模型
#     tokenized_query = list(jieba.cut_for_search(qs))  # 將查詢語句進行分詞
#     ans = bm25.get_top_n(tokenized_query, list(filtered_corpus), n=1)  # 根據查詢語句檢索，返回最相關的文檔，其中n為可調整項
#     if not ans:
#         logger.warning(f"查詢 '{qs}' 未找到相關文檔")
#         return None
#     a = ans[0]
#     # 找回與最佳匹配文本相對應的檔案名
#     res = [key for key, value in corpus_dict.items() if value == a]
#     if res:
#         logger.info(f"查詢 '{qs}' 檢索成功，相關檔案ID: {res[0]}")
#         return res[0]  # 回傳檔案名
#     else:
#         logger.warning(f"查詢 '{qs}' 找到相關文檔，但無法匹配檔案ID")
#         return None


# if __name__ == "__main__":
#     # 使用argparse解析命令列參數
#     parser = argparse.ArgumentParser(description='處理路徑和文件的程序。')
#     parser.add_argument('--question_path', type=str, required=True, help='讀取發布題目路徑')  # 問題文件的路徑
#     parser.add_argument('--source_path', type=str, required=True, help='讀取參考資料路徑')  # 參考資料的路徑
#     parser.add_argument('--output_path', type=str, required=True, help='輸出符合參賽格式的答案路徑')  # 答案輸出的路徑

#     args = parser.parse_args()  # 解析參數

#     logger.info("程序開始運行")

#     answer_dict = {"answers": []}  # 初始化字典

#     try:
#         with open(args.question_path, 'rb') as f:
#             qs_ref = json.load(f)  # 讀取問題檔案
#         logger.info(f"成功讀取問題文件: {args.question_path}")
#     except Exception as e:
#         logger.error(f"讀取問題文件 {args.question_path} 時出錯: {e}")
#         raise

#     # 載入保險相關資料
#     source_path_insurance = os.path.join(args.source_path, 'insurance')  # 設定參考資料路徑
#     corpus_dict_insurance = load_data(source_path_insurance)

#     # 載入金融相關資料
#     source_path_finance = os.path.join(args.source_path, 'finance')  # 設定參考資料路徑
#     corpus_dict_finance = load_data(source_path_finance)

#     # 讀取FAQ的pid映射
#     try:
#         with open(os.path.join(args.source_path, 'faq/pid_map_content.json'), 'rb') as f_s:
#             key_to_source_dict = json.load(f_s)  # 讀取參考資料文件
#             key_to_source_dict = {int(key): value for key, value in key_to_source_dict.items()}
#         logger.info("成功讀取FAQ的pid_map_content.json")
#     except Exception as e:
#         logger.error(f"讀取FAQ的pid_map_content.json時出錯: {e}")
#         raise

#     # 處理每個問題
#     for q_dict in qs_ref.get('questions', []):
#         qid = q_dict.get('qid')
#         category = q_dict.get('category')
#         query = q_dict.get('query')
#         source = q_dict.get('source')

#         if not all([qid, category, query, source]):
#             logger.warning(f"問題條目不完整，跳過: {q_dict}")
#             continue

#         try:
#             if category == 'finance':
#                 # 進行檢索
#                 retrieved = BM25_retrieve(query, source, corpus_dict_finance)
#                 logger.debug(f"Finance類別問題QID {qid} 檢索結果: {retrieved}")
#                 answer_dict['answers'].append({"qid": qid, "retrieve": retrieved})

#             elif category == 'insurance':
#                 retrieved = BM25_retrieve(query, source, corpus_dict_insurance)
#                 logger.debug(f"Insurance類別問題QID {qid} 檢索結果: {retrieved}")
#                 answer_dict['answers'].append({"qid": qid, "retrieve": retrieved})

#             elif category == 'faq':
#                 corpus_dict_faq = {key: str(value) for key, value in key_to_source_dict.items() if key in source}
#                 retrieved = BM25_retrieve(query, source, corpus_dict_faq)
#                 logger.debug(f"FAQ類別問題QID {qid} 檢索結果: {retrieved}")
#                 answer_dict['answers'].append({"qid": qid, "retrieve": retrieved})

#             else:
#                 logger.error(f"未知的分類類別: {category}，QID: {qid}")
#                 raise ValueError("Unknown category")
#         except Exception as e:
#             logger.error(f"處理QID {qid} 時出錯: {e}")

#     # 將答案字典保存為json文件
#     try:
#         with open(args.output_path, 'w', encoding='utf8') as f:
#             json.dump(answer_dict, f, ensure_ascii=False, indent=4)  # 儲存檔案，確保格式和非ASCII字符
#         logger.info(f"成功將答案保存到: {args.output_path}")
#     except Exception as e:
#         logger.error(f"將答案保存到 {args.output_path} 時出錯: {e}")
#         raise

#     logger.info("程序運行完成")

#------------------------------------------------------------------------------------------------------------------------

import os
import json
import argparse
import logging

from tqdm import tqdm
import jieba  # 用於中文文本分詞
import pdfplumber  # 用於從PDF文件中提取文字的工具
from rank_bm25 import BM25Okapi  # 使用BM25演算法進行文件檢索

import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

# 配置日誌設置
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 創建控制台處理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 創建文件處理器
file_handler = logging.FileHandler('pdf_extraction.log', mode='w', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)

# 創建日誌格式
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# 添加處理器到日誌器
logger.addHandler(console_handler)
logger.addHandler(file_handler)


# 讀取單個PDF文件並返回其文本內容，加入OCR功能
def read_pdf(pdf_loc, page_infos: list = None):
    logger.debug(f"打開PDF文件: {pdf_loc}")
    try:
        pdf = pdfplumber.open(pdf_loc)  # 打開指定的PDF文件
    except Exception as e:
        logger.error(f"無法打開PDF文件 {pdf_loc}: {e}")
        raise

    try:
        # 如果指定了頁面範圍，則只提取該範圍的頁面，否則提取所有頁面
        pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
        pdf_text = ''
        for idx, page in enumerate(pages):
            text = page.extract_text()  # 提取頁面的文本內容
            if text:
                pdf_text += text + "\n\n"
                logger.debug(f"成功從文件 {pdf_loc} 的第 {idx + 1} 頁提取文本。")
            else:
                logger.warning(f"頁面 {idx + 1} 無法提取文本，嘗試使用OCR。")
                try:
                    # 將頁面轉換為圖像
                    image = page.to_image(resolution=300).original
                    # 使用PIL將圖像轉換為RGB模式
                    pil_image = image.convert("RGB")
                    # 使用pytesseract進行OCR
                    ocr_text = pytesseract.image_to_string(pil_image, lang='chi_tra')  # 使用繁體中文語言包
                    if ocr_text.strip():
                        pdf_text += ocr_text + "\n\n"
                        logger.debug(f"成功從文件 {pdf_loc} 的第 {idx + 1} 頁使用OCR提取文本。")
                    else:
                        logger.warning(f"頁面 {idx + 1} 的OCR提取失敗。")
                except Exception as e:
                    logger.error(f"頁面 {idx + 1} 使用OCR時出錯: {e}")
        logger.debug(f"完成提取PDF文件: {pdf_loc}")
        return pdf_text  # 返回萃取出的文本
    except Exception as e:
        logger.error(f"提取PDF文件 {pdf_loc} 時出錯: {e}")
        raise
    finally:
        pdf.close()  # 關閉PDF文件
        logger.debug(f"已關閉PDF文件: {pdf_loc}")


# 載入參考資料，返回一個字典，key為檔案ID，value為PDF檔內容的文本
def load_data(source_path):
    logger.info(f"開始載入資料夾: {source_path}")
    try:
        masked_file_ls = os.listdir(source_path)  # 獲取資料夾中的檔案列表
    except Exception as e:
        logger.error(f"無法列出目錄 {source_path}: {e}")
        raise

    corpus_dict = {}
    for file in tqdm(masked_file_ls, desc="載入PDF文件"):
        if not file.lower().endswith('.pdf'):
            logger.warning(f"跳過非PDF文件: {file}")
            continue
        try:
            file_id = int(file.replace('.pdf', ''))
        except ValueError:
            logger.warning(f"跳過命名不正確的文件 (無法轉換為整數ID): {file}")
            continue
        file_path = os.path.join(source_path, file)
        logger.debug(f"處理文件: {file_path}")
        try:
            text = read_pdf(file_path)
            corpus_dict[file_id] = text
            logger.info(f"成功提取文件 {file} 的文本內容")
        except Exception as e:
            logger.error(f"提取文件 {file} 時出錯: {e}")
    logger.info(f"完成載入資料夾: {source_path}")
    return corpus_dict


# 根據查詢語句和指定的來源，檢索答案
def BM25_retrieve(qs, source, corpus_dict):
    logger.debug(f"開始BM25檢索，查詢: {qs}, 來源: {source}")
    try:
        filtered_corpus = [corpus_dict[int(file_id)] for file_id in source]
    except KeyError as e:
        logger.error(f"來源中存在無效的檔案ID: {e}")
        raise

    # [TODO] 可自行替換其他檢索方式，以提升效能

    tokenized_corpus = [list(jieba.cut_for_search(doc)) for doc in filtered_corpus]  # 將每篇文檔進行分詞
    bm25 = BM25Okapi(tokenized_corpus)  # 使用BM25演算法建立檢索模型
    tokenized_query = list(jieba.cut_for_search(qs))  # 將查詢語句進行分詞
    ans = bm25.get_top_n(tokenized_query, list(filtered_corpus), n=1)  # 根據查詢語句檢索，返回最相關的文檔，其中n為可調整項
    if not ans:
        logger.warning(f"查詢 '{qs}' 未找到相關文檔")
        return None
    a = ans[0]
    # 找回與最佳匹配文本相對應的檔案ID
    res = [key for key, value in corpus_dict.items() if a in value]
    if res:
        logger.info(f"查詢 '{qs}' 檢索成功，相關檔案ID: {res[0]}")
        return res[0]  # 回傳檔案ID
    else:
        logger.warning(f"查詢 '{qs}' 找到相關文檔，但無法匹配檔案ID")
        return None


if __name__ == "__main__":
    # 使用argparse解析命令列參數
    parser = argparse.ArgumentParser(description='處理路徑和文件的程序。')
    parser.add_argument('--question_path', type=str, required=True, help='讀取發布題目路徑')  # 問題文件的路徑
    parser.add_argument('--source_path', type=str, required=True, help='讀取參考資料路徑')  # 參考資料的路徑
    parser.add_argument('--output_path', type=str, required=True, help='輸出符合參賽格式的答案路徑')  # 答案輸出的路徑

    args = parser.parse_args()  # 解析參數

    logger.info("程序開始運行")

    answer_dict = {"answers": []}  # 初始化字典

    # 讀取問題檔案
    try:
        with open(args.question_path, 'rb') as f:
            qs_ref = json.load(f)  # 讀取問題檔案
        logger.info(f"成功讀取問題文件: {args.question_path}")
    except Exception as e:
        logger.error(f"讀取問題文件 {args.question_path} 時出錯: {e}")
        raise

    # 載入保險相關資料
    source_path_insurance = os.path.join(args.source_path, 'insurance')  # 設定參考資料路徑
    corpus_dict_insurance = load_data(source_path_insurance)

    # 載入金融相關資料
    source_path_finance = os.path.join(args.source_path, 'finance')  # 設定參考資料路徑
    corpus_dict_finance = load_data(source_path_finance)

    # 讀取FAQ的pid映射
    try:
        with open(os.path.join(args.source_path, 'faq/pid_map_content.json'), 'rb') as f_s:
            key_to_source_dict = json.load(f_s)  # 讀取參考資料文件
            key_to_source_dict = {int(key): value for key, value in key_to_source_dict.items()}
        logger.info("成功讀取FAQ的pid_map_content.json")
    except Exception as e:
        logger.error(f"讀取FAQ的pid_map_content.json時出錯: {e}")
        raise

    # 處理每個問題
    for q_dict in qs_ref.get('questions', []):
        qid = q_dict.get('qid')
        category = q_dict.get('category')
        query = q_dict.get('query')
        source = q_dict.get('source')

        if not all([qid, category, query, source]):
            logger.warning(f"問題條目不完整，跳過: {q_dict}")
            continue

        try:
            if category == 'finance':
                # 進行檢索
                retrieved = BM25_retrieve(query, source, corpus_dict_finance)
                logger.debug(f"Finance類別問題QID {qid} 檢索結果: {retrieved}")
                answer_dict['answers'].append({"qid": qid, "retrieve": retrieved})

            elif category == 'insurance':
                retrieved = BM25_retrieve(query, source, corpus_dict_insurance)
                logger.debug(f"Insurance類別問題QID {qid} 檢索結果: {retrieved}")
                answer_dict['answers'].append({"qid": qid, "retrieve": retrieved})

            elif category == 'faq':
                # 合併 FAQ 的內容為單一字符串
                corpus_dict_faq = {key: str(value) for key, value in key_to_source_dict.items() if key in source}
                retrieved = BM25_retrieve(query, source, corpus_dict_faq)
                logger.debug(f"FAQ類別問題QID {qid} 檢索結果: {retrieved}")
                answer_dict['answers'].append({"qid": qid, "retrieve": retrieved})

            else:
                logger.error(f"未知的分類類別: {category}，QID: {qid}")
                raise ValueError("Unknown category")
        except Exception as e:
            logger.error(f"處理QID {qid} 時出錯: {e}")

    # 將答案字典保存為json文件
    try:
        with open(args.output_path, 'w', encoding='utf8') as f:
            json.dump(answer_dict, f, ensure_ascii=False, indent=4)  # 儲存檔案，確保格式和非ASCII字符
        logger.info(f"成功將答案保存到: {args.output_path}")
    except Exception as e:
        logger.error(f"將答案保存到 {args.output_path} 時出錯: {e}")
        raise

    logger.info("程序運行完成")

'''
python test1_retrieve.py \
    --question_path /Users/danny9199/Downloads/玉山AI競賽/競賽資料集/dataset/preliminary/questions_example.json \
    --source_path /Users/danny9199/Downloads/玉山AI競賽/競賽資料集/reference \
    --output_path /Users/danny9199/Downloads/玉山AI競賽/競賽資料集/dataset/preliminary/pred_retrieve1.json
'''
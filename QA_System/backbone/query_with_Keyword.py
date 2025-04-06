import json
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk

nltk.download('punkt')

class DocumentProcessor:
    @staticmethod
    def tokenize(text):
        return word_tokenize(text.lower())

def load_documents(processed_path):
    doc_ids = []
    docs = []
    with open(processed_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            doc_ids.append(item["doc_id"])
            docs.append(item["text"])
    return doc_ids, docs

def load_questions(val_path):
    questions = []
    with open(val_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            questions.append(item["question"])
    return questions

def normalize_scores(scores):
    min_s = np.min(scores)
    max_s = np.max(scores)
    return (scores - min_s) / (max_s - min_s + 1e-8)

def predict_top_document(query, doc_ids, docs, tfidf_vectorizer, tfidf_matrix, bm25, method="bm25", alpha=0.5, top_k=5):
    tokens = DocumentProcessor.tokenize(query)

    if method == "bm25":
        scores = bm25.get_scores(tokens)

    elif method == "tfidf":
        query_vec = tfidf_vectorizer.transform([query])
        scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    elif method == "hybrid":
        bm25_scores = bm25.get_scores(tokens)
        query_vec = tfidf_vectorizer.transform([query])
        tfidf_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

        norm_bm25 = normalize_scores(bm25_scores)
        norm_tfidf = normalize_scores(tfidf_scores)

        scores = alpha * norm_bm25 + (1 - alpha) * norm_tfidf

    else:
        raise ValueError("Invalid method: choose 'bm25', 'tfidf', or 'hybrid'")

    top_k_indices = np.argsort(scores)[::-1][:top_k]
    results = [(idx, doc_ids[idx], docs[idx]) for idx in top_k_indices]

    return results

def extract_answer_from_doc(doc_text, max_words=2):
    sentences = sent_tokenize(doc_text)
    if not sentences:
        return ""
    first_sentence = sentences[0]
    tokens = word_tokenize(first_sentence)
    return " ".join(tokens[:max_words])

if __name__ == "__main__":
    # === 配置区域 ===
    '''
    mode=1
    input:.josnl文件
    output:.jsonl文件

    mode=2
    input:手动输入查询
    output:"answer": answer,
            "document_id": doc_id_list
    '''
    method = "hybrid"  # 选择检索方法："bm25", "tfidf", "hybrid"
    mode = "2"         # 选择模式："1"=识别验证集格式的文件批量预测并输出.jsonl文件，"2"=手动输入单个查询问题
    query = "when did the 1st world war officially end"  # mode = "2" 时使用的查询
    top_k = 5
    max_words = 2

    '''hybrid'''
    ###现在试了一下按照固定比例，后续调用tfidf/bm25的方式参考predict_top_document函数，写在else if后面或者新开一个函数。
    hybrid_alpha = 0.7  # hybrid 模式下 BM25 与 TF-IDF 的加权比例

    # === 路径设置 ===
    base_dir = "/Users/cusnsheep/Documents/dsa_course/S2/COMP5423/project/data_and_code(1)/data/output/"
    doc_path = base_dir + "processed_plain.jsonl"
    bm25_path = base_dir + "plain_bm25.pkl"
    tfidf_path = base_dir + "plain_tfidf.pkl"
    val_path = "/Users/cusnsheep/Documents/dsa_course/S2/COMP5423/project/data_and_code(1)/data/val.jsonl"
    output_path = base_dir + f"validation_prediction_{method}.jsonl"

    # === 加载文档和模型 ===
    doc_ids, docs = load_documents(doc_path)

    with open(bm25_path, 'rb') as f:
        bm25 = pickle.load(f)

    if method in ["tfidf", "hybrid"]:
        with open(tfidf_path, 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
            tfidf_matrix = tfidf_vectorizer.transform(docs)
    else:
        tfidf_vectorizer = None
        tfidf_matrix = None

    if mode.strip() == "1":
        # === 验证集批量预测 ===
        questions = load_questions(val_path)
        with open(output_path, 'w', encoding='utf-8') as f_out:
            for question in questions:
                top_docs = predict_top_document(
                    question, doc_ids, docs,
                    tfidf_vectorizer=tfidf_vectorizer,
                    tfidf_matrix=tfidf_matrix,
                    bm25=bm25,
                    method=method,
                    alpha=hybrid_alpha,
                    top_k=top_k
                )
                doc_id_list = [doc_id for (_, doc_id, _) in top_docs]
                answer = extract_answer_from_doc(top_docs[0][2], max_words=max_words) if top_docs else ""

                result = {
                    "question": question,
                    "answer": answer,
                    "document_id": doc_id_list
                }
                f_out.write(json.dumps(result, ensure_ascii=False) + '\n')

        print(f"\n{method.upper()} 检索预测结果已保存至：{output_path}")

    elif mode.strip() == "2":
        # === 手动查询预测（输出 JSON） ===
        top_docs = predict_top_document(
            query, doc_ids, docs,
            tfidf_vectorizer=tfidf_vectorizer,
            tfidf_matrix=tfidf_matrix,
            bm25=bm25,
            method=method,
            alpha=hybrid_alpha,
            top_k=top_k
        )

        doc_id_list = [doc_id for (_, doc_id, _) in top_docs]
        answer = extract_answer_from_doc(top_docs[0][2], max_words=max_words) if top_docs else ""

        result = {
            "question": query,
            "answer": answer,
            "document_id": doc_id_list
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))

    else:
        print("无效的模式选择，请设置 mode 为 '1' 或 '2'")

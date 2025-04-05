import json
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
from tqdm import tqdm
from utils import log_message, debug, info, warning, error
import os
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi

# 确保NLTK资源仅初始化一次，添加错误处理
try:
    # 设置NLTK数据目录到项目内部，避免权限问题
    nltk_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache", "nltk_data")
    os.makedirs(nltk_data_dir, exist_ok=True)
    nltk.data.path.append(nltk_data_dir)
    
    # 检查是否已经下载了punkt
    if not os.path.exists(os.path.join(nltk_data_dir, "tokenizers", "punkt")):
        info("正在下载NLTK punkt资源，这可能需要一点时间...")
        nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
        info("NLTK punkt资源下载完成")
    else:
        debug("已存在NLTK punkt资源")
except Exception as e:
    error(f"初始化NLTK时出错: {str(e)}")
    # 失败后尝试使用简单的分词方法
    warning("将使用备用分词方法")

class DocumentProcessor:
    @staticmethod
    def tokenize(text):
        try:
            return word_tokenize(text.lower())
        except Exception as e:
            warning(f"NLTK分词失败: {str(e)}，使用备用分词方法")
            # 备用分词方法
            return text.lower().split()


class BaseSearchEngine:
    """搜索引擎基类，包含通用功能"""
    
    @staticmethod
    def load_documents(processed_path):
        """
        加载文档数据
        
        参数:
            processed_path: 处理后的文档路径
            
        返回:
            doc_ids: 文档ID列表
            docs: 文档内容列表
        """
        doc_ids = []
        docs = []
        with open(processed_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                doc_ids.append(item["doc_id"])
                docs.append(item["text"])
        return doc_ids, docs
    
    @staticmethod
    def load_questions(val_path):
        """
        加载问题数据
        
        参数:
            val_path: 验证集路径
            
        返回:
            questions: 问题列表 (每个元素是一个包含问题信息的字典)
        """
        questions = []
        with open(val_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                questions.append(item)  # 保留整个问题项目
        return questions
    
    @staticmethod
    def load_bm25(model_path):
        """
        加载BM25模型
        
        参数:
            model_path: BM25模型路径
            
        返回:
            bm25: 加载的BM25模型
        """
        with open(model_path, 'rb') as f:
            bm25 = pickle.load(f)
        return bm25
    
    @staticmethod
    def load_tfidf(model_path):
        """
        加载TF-IDF模型
        
        参数:
            model_path: TF-IDF模型路径
            
        返回:
            tfidf_vectorizer: TF-IDF向量化器
            tfidf_matrix: 已经生成的TF-IDF矩阵，若未生成则为None
        """
        with open(model_path, 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        return tfidf_vectorizer, None
    
    @staticmethod
    def normalize_scores(scores):
        """
        归一化得分
        
        参数:
            scores: 原始得分数组
            
        返回:
            归一化后的得分
        """
        min_s = np.min(scores)
        max_s = np.max(scores)
        return (scores - min_s) / (max_s - min_s + 1e-8)
    
    @staticmethod
    def extract_answer_from_doc(doc_text, max_words=2):
        """
        从文档中提取答案
        
        参数:
            doc_text: 文档文本
            max_words: 最多提取的单词数
            
        返回:
            提取的答案文本
        """
        sentences = sent_tokenize(doc_text)
        if not sentences:
            return ""
        first_sentence = sentences[0]
        tokens = word_tokenize(first_sentence)
        return " ".join(tokens[:max_words])
    
    def predict_top_document(self, query, doc_ids, docs, tfidf_vectorizer, tfidf_matrix, bm25, method="bm25", alpha=0.5, top_k=5, return_format="results"):
        """
        预测与查询最相关的文档
        
        参数:
            query: 查询字符串
            doc_ids: 文档ID列表
            docs: 文档内容列表
            tfidf_vectorizer: TF-IDF向量化器
            tfidf_matrix: TF-IDF矩阵
            bm25: BM25模型
            method: 使用的方法 (bm25, tfidf, hybrid)
            alpha: hybrid模式下BM25的权重
            top_k: 返回的顶部结果数量
            return_format: 返回格式，可选值为:
                - "results": 返回三元组列表[(idx, doc_id, doc)]
                - "lists": 返回两个列表(doc_ids, scores)
            
        返回:
            根据return_format参数返回不同格式:
            - "results": 最相关的文档列表，每个元素为(idx, doc_id, doc)三元组
            - "lists": 两个列表，(doc_ids, scores)
        """
        debug(f"使用 {method} 方法检索与查询相关的文档")
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

            norm_bm25 = self.normalize_scores(bm25_scores)
            norm_tfidf = self.normalize_scores(tfidf_scores)

            scores = alpha * norm_bm25 + (1 - alpha) * norm_tfidf

        else:
            raise ValueError("Invalid method: choose 'bm25', 'tfidf', or 'hybrid'")

        top_k_indices = np.argsort(scores)[::-1][:top_k]
        
        if return_format == "results":
            results = [(idx, doc_ids[idx], docs[idx]) for idx in top_k_indices]
            return results
        elif return_format == "lists":
            top_doc_ids = [doc_ids[idx] for idx in top_k_indices]
            top_scores = [float(scores[idx]) for idx in top_k_indices]  # 转换为普通float，便于JSON序列化
            return top_doc_ids, top_scores
        else:
            raise ValueError("Invalid return_format: choose 'results' or 'lists'")


class ManualQuerySearch(BaseSearchEngine):
    """处理手动查询的类，接收webui参数进行搜索"""
    
    def __init__(self, config):
        """
        初始化查询搜索器
        
        参数:
            config: 配置字典，需包含以下键:
                - doc_path: 处理后的文档路径
                - bm25_path: BM25模型路径
                - tfidf_path: TF-IDF模型路径
                - method: 搜索方法 (bm25, tfidf, hybrid)
                - hybrid_alpha: hybrid模式下的BM25权重
                - top_k: 返回的顶部结果数量
                - max_words: 提取作为答案的最大单词数
        """
        self.config = config
        self.method = config.get("method", "bm25")
        self.hybrid_alpha = config.get("hybrid_alpha", 0.7)
        self.top_k = config.get("top_k", 5)
        self.max_words = config.get("max_words", 2)
        
        try:
            # 加载文档
            doc_path = config.get("doc_path")
            info(f"正在加载文档: {doc_path}")
            self.doc_ids, self.docs = self.load_documents(doc_path)
            info(f"成功加载 {len(self.doc_ids)} 个文档")
            
            # 加载BM25模型
            bm25_path = config.get("bm25_path")
            info(f"正在加载BM25模型: {bm25_path}")
            self.bm25 = self.load_bm25(bm25_path)
            info("BM25模型加载成功")
            
            # 如果需要，加载TF-IDF模型
            if self.method in ["tfidf", "hybrid"]:
                tfidf_path = config.get("tfidf_path")
                info(f"正在加载TF-IDF模型: {tfidf_path}")
                self.tfidf_vectorizer, _ = self.load_tfidf(tfidf_path)
                    
                # 在需要时再生成TF-IDF矩阵，避免一开始就占用大量内存
                info("TF-IDF向量器加载成功，将在首次查询时生成矩阵")
                self._tfidf_matrix = None
            else:
                self.tfidf_vectorizer = None
                self._tfidf_matrix = None
                
            info(f"ManualQuerySearch初始化完成，使用方法: {self.method}")
        except Exception as e:
            error(f"初始化ManualQuerySearch时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    @property
    def tfidf_matrix(self):
        """延迟加载TF-IDF矩阵，只在需要时才生成"""
        if self._tfidf_matrix is None and self.tfidf_vectorizer is not None:
            info("正在生成TF-IDF矩阵，这可能需要一点时间...")
            self._tfidf_matrix = self.tfidf_vectorizer.transform(self.docs)
            info(f"TF-IDF矩阵生成完成，形状: {self._tfidf_matrix.shape}")
        return self._tfidf_matrix
    
    def search(self, query):
        """
        使用初始化的模型搜索查询
        
        参数:
            query: 查询字符串
            
        返回:
            包含answer和document_id的字典
        """
        debug(f"处理查询: '{query}'")
        
        try:
            # 执行文档检索
            results = self.predict_top_document(
                query, self.doc_ids, self.docs,
                tfidf_vectorizer=self.tfidf_vectorizer,
                tfidf_matrix=self.tfidf_matrix,
                bm25=self.bm25,
                method=self.method,
                alpha=self.hybrid_alpha,
                top_k=self.top_k,
                return_format="results"
            )
            
            # 解析结果
            if results and len(results) > 0:
                # 提取文档ID列表，并确保都是字符串类型
                doc_id_list = [str(doc_id) for (_, doc_id, _) in results]
                
                # 从第一个文档提取答案
                first_doc_text = results[0][2]
                answer = self.extract_answer_from_doc(first_doc_text, max_words=self.max_words)
                
                debug(f"检索结果: 找到 {len(doc_id_list)} 个相关文档")
                return {
                    "question": query,
                    "answer": answer,
                    "document_id": doc_id_list
                }
            else:
                # 未找到相关文档
                debug("检索结果: 未找到相关文档")
                return {
                    "question": query,
                    "answer": "未找到相关信息",
                    "document_id": []
                }
                
        except Exception as e:
            error(f"搜索过程中出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "question": query,
                "answer": f"搜索过程中出错: {str(e)}",
                "document_id": []
            }


class BatchEvaluator(BaseSearchEngine):
    """批量评估类，用于评估检索方法在验证集上的性能"""
    
    def __init__(self, config):
        """
        初始化批量评估类
        
        参数:
            config: 配置字典，包含评估所需的参数
        """
        super().__init__()
        self.method = config.get("method", "hybrid")  # 检索方法: hybrid/bm25/tfidf
        self.hybrid_alpha = config.get("hybrid_alpha", 0.7)  # hybrid方法中BM25的权重
        self.top_k = config.get("top_k", 5)  # 返回的文档数量
        self.max_words = config.get("max_words", 2)  # 截断文档标题的最大单词数
        
        self.doc_path = config.get("doc_path")  # 文档路径
        self.bm25_path = config.get("bm25_path")  # BM25模型路径
        self.tfidf_path = config.get("tfidf_path")  # TF-IDF模型路径
        self.val_path = config.get("val_path")  # 验证集路径
        self.output_path = config.get("output_path")  # 输出路径
        
        info(f"[BatchEvaluator] 初始化评估器，方法: {self.method}, 混合比例: {self.hybrid_alpha}, Top-K: {self.top_k}")
        info(f"[BatchEvaluator] 数据路径: 文档={self.doc_path}, 验证集={self.val_path}")
        debug(f"[BatchEvaluator] 模型路径: BM25={self.bm25_path}, TF-IDF={self.tfidf_path}")
        info(f"[BatchEvaluator] 输出路径: {self.output_path}")
    
    def evaluate(self):
        """
        在验证集上评估检索方法

        返回:
            输出文件路径
        """
        info(f"[BatchEvaluator] 开始评估过程，方法: {self.method}")
        
        # 加载文档
        debug(f"[BatchEvaluator] 开始加载文档: {self.doc_path}")
        doc_ids, docs = self.load_documents(self.doc_path)
        info(f"[BatchEvaluator] 成功加载 {len(doc_ids)} 个文档")
        
        # 加载BM25和TF-IDF模型
        debug(f"[BatchEvaluator] 开始加载BM25模型: {self.bm25_path}")
        bm25 = self.load_bm25(self.bm25_path)
        debug(f"[BatchEvaluator] BM25模型加载成功")
        
        debug(f"[BatchEvaluator] 开始加载TF-IDF模型: {self.tfidf_path}")
        tfidf_vectorizer, _ = self.load_tfidf(self.tfidf_path)
        # 生成TF-IDF矩阵
        debug(f"[BatchEvaluator] 开始生成TF-IDF矩阵")
        tfidf_matrix = tfidf_vectorizer.transform(docs)
        debug(f"[BatchEvaluator] TF-IDF模型加载成功，矩阵形状: {tfidf_matrix.shape}")
        
        # 加载验证集问题
        debug(f"[BatchEvaluator] 开始加载验证集问题: {self.val_path}")
        questions = self.load_questions(self.val_path)
        info(f"[BatchEvaluator] 成功加载 {len(questions)} 个问题")
        
        # 存储评估结果
        info(f"[BatchEvaluator] 开始评估，总共 {len(questions)} 个问题")
        results = []
        empty_results = 0
        total_pred_docs = 0
        
        # 评估每个问题
        start_time = time.time()
        for i, question_item in enumerate(questions):
            # 每10%打印一次进度
            if (i+1) % max(1, len(questions)//10) == 0 or i+1 == len(questions):
                progress = (i+1) / len(questions) * 100
                elapsed = time.time() - start_time
                eta = (elapsed / (i+1)) * (len(questions) - i - 1) if i > 0 else 0
                info(f"[BatchEvaluator] 进度: {progress:.1f}% ({i+1}/{len(questions)}), 已用时间: {elapsed:.1f}秒, 预计剩余: {eta:.1f}秒")
            
            # 从问题项目中获取查询文本和参考文档ID
            query = question_item["question"]
            ref_doc_id = question_item.get("document_id", "")
            
            # 预测文档
            pred_doc_ids, pred_scores = self.predict_top_document(
                query, doc_ids, docs, tfidf_vectorizer, tfidf_matrix, bm25,
                method=self.method, alpha=self.hybrid_alpha, top_k=self.top_k,
                return_format="lists"
            )
            
            if not pred_doc_ids:
                empty_results += 1
                warning(f"[BatchEvaluator] 警告: 问题 {i+1} 未找到匹配文档: {query[:50]}...")
            
            total_pred_docs += len(pred_doc_ids)
            
            # 记录结果
            result = {
                "question": query,
                "ref_doc_id": ref_doc_id,
                "pred_doc_ids": pred_doc_ids,
                "pred_scores": pred_scores
            }
            results.append(result)
        
        # 计算统计信息
        avg_docs = total_pred_docs / len(questions) if questions else 0
        info(f"[BatchEvaluator] 评估完成，总问题数: {len(questions)}, 空结果数: {empty_results}, 平均文档数: {avg_docs:.2f}")
        
        # 保存结果
        debug(f"[BatchEvaluator] 开始保存评估结果到: {self.output_path}")
        with open(self.output_path, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        info(f"[BatchEvaluator] 评估结果成功保存到: {self.output_path}")
        
        return self.output_path

'''
if __name__ == "__main__":
    # === 配置区域 ===

    mode=1: 批量评估
    input:.josnl文件
    output:.jsonl文件

    mode=2: 手动查询
    input:手动输入查询
    output:"answer": answer,
            "document_id": doc_id_list

    mode = "2"  # 选择模式："1"=批量评估，"2"=手动查询
    
    # 基础配置
    config = {
        "method": "hybrid",           # 选择检索方法："bm25", "tfidf", "hybrid"
        "hybrid_alpha": 0.7,          # hybrid 模式下 BM25 权重
        "top_k": 5,                   # 返回的顶部结果数量
        "max_words": 2,               # 提取作为答案的最大单词数
        
        # 路径设置
        "base_dir": "/Users/cusnsheep/Documents/dsa_course/S2/COMP5423/project/data_and_code(1)/data/output/",
        "doc_path": "/Users/cusnsheep/Documents/dsa_course/S2/COMP5423/project/data_and_code(1)/data/output/processed_plain.jsonl",
        "bm25_path": "/Users/cusnsheep/Documents/dsa_course/S2/COMP5423/project/data_and_code(1)/data/output/plain_bm25.pkl",
        "tfidf_path": "/Users/cusnsheep/Documents/dsa_course/S2/COMP5423/project/data_and_code(1)/data/output/plain_tfidf.pkl",
        "val_path": "/Users/cusnsheep/Documents/dsa_course/S2/COMP5423/project/data_and_code(1)/data/val.jsonl",
        "output_path": "/Users/cusnsheep/Documents/dsa_course/S2/COMP5423/project/data_and_code(1)/data/output/validation_prediction_hybrid.jsonl"
    }
    
    if mode.strip() == "1":
        # 批量评估模式
        evaluator = BatchEvaluator(config)
        evaluator.evaluate()
        
    elif mode.strip() == "2":
        # 手动查询模式
        query = "when did the 1st world war officially end"  # 测试查询
        
        search = ManualQuerySearch(config)
        result = search.search(query)
        
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
    else:
        print("无效的模式选择，请设置 mode 为 '1' 或 '2'")

'''
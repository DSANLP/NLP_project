import json
import re
import random
import os
from typing import List, Dict, Any, Tuple
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import pickle
from bs4 import BeautifulSoup
import html2text
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk
import logging

# 确保nltk数据已下载
try:
    nltk.download('punkt', quiet=True)
except:
    logging.warning("无法下载NLTK punkt数据，请确保已手动安装")

class DocumentProcessor:
    """处理文档的基类，包含通用的文本处理方法"""
    
    @staticmethod
    def clean_wikipedia_text(text):
        """文本的清理"""
        # 首先去除换行符
        text = re.sub(r'\n', ' ', text)
        # 移除Contents (hide)部分及其周围空格
        text = re.sub(r'\s*Contents \( hide \)\s*', ' ', text)
        # 移除(edit)标记及其周围空格
        text = re.sub(r'\s*\(edit\)\s*', ' ', text)
        # 专门处理括号内有空格的( edit )标记
        text = re.sub(r'\(\sedit\s\)', '', text)  # 匹配( edit )
        text = re.sub(r'\(\sedit\)', '', text)    # 匹配(edit )
        text = re.sub(r'\(edit\s\)', '', text)    # 匹配( edit)
       
        # 移除其他维基百科特有格式
        text = re.sub(r'Jump to : navigation , search', '', text)
        text = re.sub(r'Categories :.*', '', text)
        text = re.sub(r'Hidden categories :.*', '', text)
        text = re.sub(r'Edit links', '', text)
        text = re.sub(r'Retrieved from .*', '', text)
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'See also .*', '', text, flags=re.DOTALL)
        text = re.sub(r'References .*', '', text, flags=re.DOTALL)
        # 清理多余的空格
        text = re.sub(r' +', ' ', text)
        return text.strip()

    @staticmethod
    def tokenize(text):
        """基本分词处理"""
        return word_tokenize(text.lower())


class PlainTextProcessor(DocumentProcessor):
    """处理纯文本的类"""
    
    @classmethod
    def clean_html_to_plaintext(cls, html):
        """处理为纯文本"""
        soup = BeautifulSoup(html, 'html.parser')
        # 移除非内容元素
        for script in soup(["script", "style", "table", "ul", "ol", "footer", "nav"]):
            script.decompose()
        text = soup.get_text(separator=' ', strip=True)
        # 应用维基百科特定清理
        text = cls.clean_wikipedia_text(text)
        return text


class MarkdownProcessor(DocumentProcessor):
    """处理Markdown文本的类"""
    
    @classmethod
    def html_to_markdown(cls, html):
        """转换为Markdown格式"""
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = True
        h.bypass_tables = False
        h.single_line_break = True
        markdown = h.handle(html)
        # 应用维基百科特定清理
        markdown = cls.clean_wikipedia_text(markdown)
        return markdown


class DocumentIndexer:
    """文档索引类，用于构建搜索模型"""
    
    def __init__(self, corpus):
        self.corpus = corpus
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.bm25 = None
    
    def calculate_tfidf(self):
        """计算TF-IDF模型"""
        logging.info("开始计算TF-IDF模型...")
        self.tfidf_vectorizer = TfidfVectorizer(
            tokenizer=DocumentProcessor.tokenize, 
            stop_words='english'
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.corpus)
        logging.info("TF-IDF模型计算完成")
        return self.tfidf_vectorizer, self.tfidf_matrix
    
    def calculate_bm25(self):
        """计算BM25模型"""
        logging.info("开始计算BM25模型...")
        
        # 使用tqdm包装分词过程
        tokenized_corpus = []
        for doc in tqdm(self.corpus, desc="分词处理中", unit="doc"):
            tokenized_corpus.append(DocumentProcessor.tokenize(doc))
        
        logging.info("构建BM25索引中...")
        self.bm25 = BM25Okapi(tokenized_corpus)
        logging.info("BM25索引构建完成")
        return self.bm25
    
    def save_models(self, output_prefix):
        """保存模型到文件"""
        if self.tfidf_vectorizer:
            logging.info(f"正在保存TF-IDF模型到 {output_prefix}_tfidf.pkl...")
            with open(f'{output_prefix}_tfidf.pkl', 'wb') as f:
                pickle.dump(self.tfidf_vectorizer, f)
            logging.info(f"TF-IDF模型已保存")
        if self.bm25:
            logging.info(f"正在保存BM25模型到 {output_prefix}_bm25.pkl...")
            with open(f'{output_prefix}_bm25.pkl', 'wb') as f:
                pickle.dump(self.bm25, f)
            logging.info(f"BM25模型已保存")


class DocumentPipeline:
    """文档处理管道，整合整个流程"""
    
    def __init__(self, input_path, model_output_dir, data_output_dir, config=None):
        self.input_path = input_path
        # 确保目录路径以路径分隔符结尾
        self.model_output_dir = os.path.join(model_output_dir, '') if not model_output_dir.endswith(os.path.sep) else model_output_dir
        self.data_output_dir = os.path.join(data_output_dir, '') if not data_output_dir.endswith(os.path.sep) else data_output_dir
        self.processed_plain = []
        self.processed_markdown = []
        self.config = config
        
    def process_documents(self):
        """处理原始数据"""
        # 先计算文件行数以确定总进度
        with open(self.input_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)                                                                 
        
        logging.info(f"开始处理文档，共 {total_lines} 行...")
        
        # 重新打开文件进行处理
        with open(self.input_path, 'r', encoding='utf-8') as f:
            # Using tqdm to show progress bar
            for line in tqdm(f, total=total_lines, desc="处理文档中", unit="doc"):
                try:
                    doc = json.loads(line)
                    plain_text = PlainTextProcessor.clean_html_to_plaintext(doc['document_text'])
                    markdown_text = MarkdownProcessor.html_to_markdown(doc['document_text'])
                    
                    self.processed_plain.append({
                        "doc_id": doc['document_id'],
                        "text": plain_text
                    })
                    
                    self.processed_markdown.append({
                        "doc_id": doc['document_id'],
                        "text": markdown_text
                    })
                except json.JSONDecodeError as e:
                    logging.error(f"解析JSON时出错，行内容: {line[:100]}... 错误: {e}")
                except KeyError as e:
                    logging.error(f"文档中缺少关键字段: {e}")
    
    def save_processed_data(self):
        """保存处理后的数据"""
        plain_output = os.path.join(self.data_output_dir, "processed_plain.jsonl")
        markdown_output = os.path.join(self.data_output_dir, "processed_markdown.jsonl")
        
        logging.info(f"正在保存处理后的纯文本数据到 {plain_output}...")
        with open(plain_output, 'w', encoding='utf-8') as f:
            # Using tqdm to show progress bar
            for item in tqdm(self.processed_plain, desc="保存纯文本数据", unit="doc"):
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logging.info(f"正在保存处理后的markdown数据到 {markdown_output}...")
        with open(markdown_output, 'w', encoding='utf-8') as f:
            # Using tqdm to show progress bar
            for item in tqdm(self.processed_markdown, desc="保存markdown数据", unit="doc"):
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logging.info(f"纯文本数据已保存到 {plain_output}")
        logging.info(f"Markdown数据已保存到 {markdown_output}")
    
    def build_models(self):
        """构建搜索模型"""
        plain_corpus = [doc['text'] for doc in self.processed_plain]
        markdown_corpus = [doc['text'] for doc in self.processed_markdown]
        
        logging.info("开始为纯文本构建模型...")
        plain_indexer = DocumentIndexer(plain_corpus)
        plain_indexer.calculate_tfidf()
        plain_indexer.calculate_bm25()
        plain_indexer.save_models(os.path.join(self.model_output_dir, "plain"))
        
        logging.info("开始为markdown构建模型...")
        markdown_indexer = DocumentIndexer(markdown_corpus)
        markdown_indexer.calculate_tfidf()
        markdown_indexer.calculate_bm25()
        markdown_indexer.save_models(os.path.join(self.model_output_dir, "markdown"))
        
        logging.info("所有模型已成功构建并保存")
    
    def run(self):
        """运行整个处理流程"""
        self.process_documents()
        self.save_processed_data()
        self.build_models()


class DPRDataProcessor:
    """
    处理数据为DPR模型训练格式
    """
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        # 创建缓存目录
        self.cache_dir = os.path.join("cache", "bert_base_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.documents = {}
        self.document_chunks = {}
        self.train_data = []
        self.val_data = []
        self.tfidf_vectorizer = None
        self.chunk_vectors = None
        self.chunk_list = []
        self.chunk_id_to_idx = {}
        
    def load_documents(self) -> None:
        """加载所有文档"""
        doc_file = os.path.join(self.data_dir, "processed_plain.jsonl")
        logging.info(f"正在加载文档数据: {doc_file}")
        
        with open(doc_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                doc = json.loads(line)
                self.documents[doc["doc_id"]] = doc["text"]
        
        logging.info(f"成功加载 {len(self.documents)} 个文档")
    
    def load_qa_data(self) -> None:
        """加载训练集和验证集数据"""
        # 修改路径，从original_data目录加载数据
        original_data_dir = os.path.join(os.path.dirname(self.data_dir), "original_data")
        train_file = os.path.join(original_data_dir, "train.jsonl")
        val_file = os.path.join(original_data_dir, "val.jsonl")
        
        logging.info(f"从 {original_data_dir} 加载训练和验证数据")
        
        # 加载训练数据
        with open(train_file, 'r', encoding='utf-8') as f:
            for line in f:
                self.train_data.append(json.loads(line))
        
        # 加载验证数据
        with open(val_file, 'r', encoding='utf-8') as f:
            for line in f:
                self.val_data.append(json.loads(line))
                
        logging.info(f"成功加载 {len(self.train_data)} 条训练数据和 {len(self.val_data)} 条验证数据")
    
    def chunk_documents(self, chunk_size: int = 2048, overlap: int = 100) -> None:
        """
        将文档分割成重叠的块，利用nomic-bert-2048的长序列能力
        
        Args:
            chunk_size: 每个块的大致词数，可以设置得更大
            overlap: 块之间的重叠词数
        """
        logging.info("正在将文档分割成块...")
        
        # 检查缓存 - 现在使用cache目录
        cache_file = os.path.join(self.cache_dir, f"chunks_cache_{chunk_size}_{overlap}.pkl")
        if os.path.exists(cache_file):
            logging.info(f"从缓存加载文档块: {cache_file}")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                self.document_chunks = cache_data['document_chunks']
                self.chunk_list = cache_data['chunk_list']
                for idx, chunk in enumerate(self.chunk_list):
                    self.chunk_id_to_idx[(chunk['doc_id'], chunk['chunk_id'])] = idx
            total_chunks = sum(len(chunks) for chunks in self.document_chunks.values())
            logging.info(f"成功从缓存加载 {total_chunks} 个块")
            return
            
        for doc_id, text in tqdm(self.documents.items()):
            words = text.split()
            chunks = []
            
            # 如果文档不是特别长，可以直接使用整个文档
            if len(words) <= chunk_size:
                chunks.append({
                    "doc_id": doc_id,
                    "chunk_id": 0,
                    "text": text,
                    "start_idx": 0,
                    "end_idx": len(words)
                })
            else:
                # 对于非常长的文档，仍然需要分块，但块可以更大
                start = 0
                while start < len(words):
                    end = min(start + chunk_size, len(words))
                    chunk_text = " ".join(words[start:end])
                    chunks.append({
                        "doc_id": doc_id,
                        "chunk_id": len(chunks),
                        "text": chunk_text,
                        "start_idx": start,
                        "end_idx": end
                    })
                    
                    start += chunk_size - overlap
            
            self.document_chunks[doc_id] = chunks
            
            # 维护一个列表，方便后续处理
            for chunk in chunks:
                self.chunk_list.append(chunk)
                self.chunk_id_to_idx[(chunk['doc_id'], chunk['chunk_id'])] = len(self.chunk_list) - 1
        
        total_chunks = sum(len(chunks) for chunks in self.document_chunks.values())
        logging.info(f"成功将文档分割成 {total_chunks} 个块")
        
        # 保存缓存 - 现在使用cache目录
        logging.info(f"将文档块保存到缓存: {cache_file}")
        cache_data = {
            'document_chunks': self.document_chunks,
            'chunk_list': self.chunk_list
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
    
    def build_tfidf_index(self):
        """构建TF-IDF索引"""
        logging.info("构建TF-IDF索引...")
        
        # 检查缓存 - 现在使用cache目录
        cache_file = os.path.join(self.cache_dir, "tfidf_cache.pkl")
        if os.path.exists(cache_file):
            logging.info(f"从缓存加载TF-IDF索引: {cache_file}")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                self.tfidf_vectorizer = cache_data['vectorizer']
                self.chunk_vectors = cache_data['chunk_vectors']
            logging.info("TF-IDF索引加载完成")
            return
        
        # 收集所有块的文本
        corpus = [chunk["text"] for chunk in self.chunk_list]
        
        # 创建TF-IDF向量化器
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=50000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # 将所有文档块转换为TF-IDF向量
        start_time = time.time()
        self.chunk_vectors = self.tfidf_vectorizer.fit_transform(corpus)
        elapsed = time.time() - start_time
        logging.info(f"TF-IDF索引构建完成，用时 {elapsed:.2f} 秒")
        
        # 保存缓存 - 现在使用cache目录
        logging.info(f"将TF-IDF索引保存到缓存: {cache_file}")
        cache_data = {
            'vectorizer': self.tfidf_vectorizer,
            'chunk_vectors': self.chunk_vectors
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
    
    def find_relevant_chunk(self, doc_id: int, answer: str) -> Dict:
        """
        找到包含答案的最相关文档块
        
        Args:
            doc_id: 文档ID
            answer: 要查找的答案文本
            
        Returns:
            包含答案的文档块
        """
        if doc_id not in self.document_chunks:
            return None
            
        chunks = self.document_chunks[doc_id]
        best_chunk = None
        highest_overlap = -1
        
        for chunk in chunks:
            if answer.lower() in chunk["text"].lower():
                # 简单使用包含关系作为相关性判断
                # 实际应用中可能需要更复杂的相关性评分
                overlap_score = len(answer) / len(chunk["text"])
                if overlap_score > highest_overlap:
                    highest_overlap = overlap_score
                    best_chunk = chunk
        
        return best_chunk
    
    def mine_hard_negatives_tfidf(self, question: str, answer: str, doc_id: int, n_samples: int = 3) -> List[Dict]:
        """
        使用TF-IDF挖掘困难负样本 - 与问题相关但不包含答案的文档块
        
        Args:
            question: 问题文本
            answer: 答案文本
            doc_id: 正样本文档ID
            n_samples: 要采样的负样本数量
            
        Returns:
            困难负样本块列表
        """
        # 确保TF-IDF索引已构建
        if self.tfidf_vectorizer is None or self.chunk_vectors is None:
            self.build_tfidf_index()
        
        # 将问题转换为TF-IDF向量
        query_vector = self.tfidf_vectorizer.transform([question])
        
        # 计算问题与所有文档块的相似度
        similarities = cosine_similarity(query_vector, self.chunk_vectors).flatten()
        
        # 按相似度排序获取最相似的索引
        top_indices = similarities.argsort()[::-1]
        
        # 收集不包含答案且不是同一文档的困难负样本
        hard_negatives = []
        for idx in top_indices:
            chunk = self.chunk_list[idx]
            
            # 排除相同文档的块和包含答案的块
            if chunk["doc_id"] != doc_id and answer.lower() not in chunk["text"].lower():
                hard_negatives.append(chunk)
                if len(hard_negatives) >= n_samples:
                    break
        
        # 如果找不到足够的困难负样本，返回已找到的
        return hard_negatives
    
    def sample_negative_chunks(self, question: str, answer: str, positive_chunk: Dict, n_samples: int = 3) -> List[Dict]:
        """
        采样负样本块，优先使用困难负样本
        
        Args:
            question: 问题文本
            answer: 答案文本
            positive_chunk: 正样本块
            n_samples: 要采样的负样本数量
            
        Returns:
            负样本块列表
        """
        # 首先尝试挖掘困难负样本
        hard_negatives = self.mine_hard_negatives_tfidf(
            question=question,
            answer=answer,
            doc_id=positive_chunk["doc_id"],
            n_samples=n_samples
        )
        
        # 如果找到足够的困难负样本，直接返回
        if len(hard_negatives) >= n_samples:
            return hard_negatives[:n_samples]
        
        # 否则用随机负样本补充
        all_doc_ids = list(self.document_chunks.keys())
        needed = n_samples - len(hard_negatives)
        random_negatives = []
        
        while len(random_negatives) < needed and all_doc_ids:
            doc_id = random.choice(all_doc_ids)
            
            if doc_id != positive_chunk["doc_id"] and doc_id in self.document_chunks:
                chunks = self.document_chunks[doc_id]
                if chunks:
                    chunk = random.choice(chunks)
                    # 确保不包含答案且不在已有的困难负样本中
                    if (answer.lower() not in chunk["text"].lower() and 
                        chunk not in hard_negatives and
                        chunk not in random_negatives):
                        random_negatives.append(chunk)
            
            if len(random_negatives) >= needed:
                break
        
        return hard_negatives + random_negatives
    
    def _process_batch(self, batch_items, n_negatives: int) -> List[Dict]:
        """
        处理一批数据
        
        Args:
            batch_items: 批量数据
            n_negatives: 每个问题的负样本数量
            
        Returns:
            处理后的样本列表
        """
        batch_examples = []
        
        for item in batch_items:
            try:
                # 确保文档ID存在
                doc_id = item.get("document_id") or item.get("doc_id")
                if doc_id is None:
                    continue  # 静默跳过，不记录警告
                
                question = item["question"]
                answer = item["answer"]
                
                positive_chunk = self.find_relevant_chunk(doc_id, answer)
                if positive_chunk:  # 只处理能找到正样本的情况
                    negative_chunks = self.sample_negative_chunks(
                        question=question,
                        answer=answer,
                        positive_chunk=positive_chunk,
                        n_samples=n_negatives
                    )
                    
                    if negative_chunks and len(negative_chunks) >= n_negatives:
                        example = {
                            "question": question,
                            "answer": answer,
                            "positive_ctx": positive_chunk["text"],  # 注意：这里使用单数形式
                            "negative_ctxs": [chunk["text"] for chunk in negative_chunks]
                        }
                        batch_examples.append(example)
            except Exception as e:
                logging.debug(f"处理条目时出错: {str(e)}")  # 降低日志级别，减少输出
                continue
        
        return batch_examples
        
    def prepare_dpr_training_data(self, n_negatives: int = 3, batch_size: int = 500, max_workers: int = 4) -> Tuple[List, List]:
        """
        并行准备DPR训练数据
        
        Args:
            n_negatives: 每个问题的负样本数量
            batch_size: 批处理大小
            max_workers: 并行处理的工作进程数
            
        Returns:
            (train_examples, val_examples) 元组
        """
        logging.info("正在准备DPR训练数据...")
        
        # 确保已加载所有必要数据
        if not self.documents:
            self.load_documents()
        
        if not self.train_data or not self.val_data:
            self.load_qa_data()
            
        if not self.document_chunks:
            self.chunk_documents()
            
        if self.tfidf_vectorizer is None:
            self.build_tfidf_index()
        
        start_time = time.time()
        
        # 准备训练数据
        train_examples = []
        
        # 检查训练数据缓存 - 现在使用cache目录
        train_cache_file = os.path.join(self.cache_dir, f"train_examples_cache_{n_negatives}.pkl")
        if os.path.exists(train_cache_file):
            logging.info(f"从缓存加载训练样本: {train_cache_file}")
            with open(train_cache_file, 'rb') as f:
                train_examples = pickle.load(f)
            logging.info(f"成功加载 {len(train_examples)} 条训练样本")
        else:
            logging.info(f"并行处理训练数据，使用 {max_workers} 个进程...")
            # 分批处理训练数据
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for i in range(0, len(self.train_data), batch_size):
                    end_idx = min(i + batch_size, len(self.train_data))
                    batch = self.train_data[i:end_idx]
                    futures.append(executor.submit(self._process_batch, batch, n_negatives))
                
                # 收集结果
                for future in tqdm(as_completed(futures), total=len(futures), desc="处理训练数据批次"):
                    batch_examples = future.result()
                    train_examples.extend(batch_examples)
            
            logging.info(f"已准备 {len(train_examples)} 条训练样本")
            
            # 保存训练样本缓存 - 现在使用cache目录
            logging.info(f"将训练样本保存到缓存: {train_cache_file}")
            with open(train_cache_file, 'wb') as f:
                pickle.dump(train_examples, f)
        
        # 准备验证数据
        val_examples = []
        
        # 检查验证数据缓存 - 现在使用cache目录
        val_cache_file = os.path.join(self.cache_dir, f"val_examples_cache_{n_negatives}.pkl")
        if os.path.exists(val_cache_file):
            logging.info(f"从缓存加载验证样本: {val_cache_file}")
            with open(val_cache_file, 'rb') as f:
                val_examples = pickle.load(f)
            logging.info(f"成功加载 {len(val_examples)} 条验证样本")
        else:
            logging.info(f"处理验证数据，使用 {max_workers} 个进程...")
            # 分批处理验证数据
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for i in range(0, len(self.val_data), batch_size):
                    end_idx = min(i + batch_size, len(self.val_data))
                    batch = self.val_data[i:end_idx]
                    futures.append(executor.submit(self._process_batch, batch, n_negatives))
                
                # 收集结果
                for future in tqdm(as_completed(futures), total=len(futures), desc="处理验证数据批次"):
                    batch_examples = future.result()
                    val_examples.extend(batch_examples)
            
            logging.info(f"已准备 {len(val_examples)} 条验证样本")
            
            # 保存验证样本缓存 - 现在使用cache目录
            logging.info(f"将验证样本保存到缓存: {val_cache_file}")
            with open(val_cache_file, 'wb') as f:
                pickle.dump(val_examples, f)
        
        elapsed = time.time() - start_time
        logging.info(f"DPR训练数据准备完成，用时 {elapsed:.2f} 秒")
        
        return train_examples, val_examples
    
    def save_dpr_data(self, train_examples: List, val_examples: List) -> None:
        """
        保存DPR数据到文件
        
        Args:
            train_examples: 训练样本列表
            val_examples: 验证样本列表
        """
        train_output_file = os.path.join(self.data_dir, "dpr_train.json") 
        val_output_file = os.path.join(self.data_dir, "dpr_val.json")
        
        logging.info(f"保存 {len(train_examples)} 条训练样本到 {train_output_file}")
        with open(train_output_file, 'w', encoding='utf-8') as f:
            json.dump(train_examples, f, ensure_ascii=False, indent=2)
            
        logging.info(f"保存 {len(val_examples)} 条验证样本到 {val_output_file}")
        with open(val_output_file, 'w', encoding='utf-8') as f:
            json.dump(val_examples, f, ensure_ascii=False, indent=2)
            
        logging.info(f"DPR训练数据已保存到 {train_output_file} 和 {val_output_file}")
    
    def process_all(self, max_workers: int = 4):
        """
        执行完整的DPR数据处理流程
        
        Args:
            max_workers: 最大工作进程数
        """
        self.load_documents()
        self.load_qa_data()
        self.chunk_documents()
        self.build_tfidf_index()
        train_examples, val_examples = self.prepare_dpr_training_data(max_workers=max_workers)
        self.save_dpr_data(train_examples, val_examples)
        logging.info("DPR数据处理流程完成") 
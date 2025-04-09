import json
import re
from bs4 import BeautifulSoup
import html2text
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import pickle
from nltk.tokenize import word_tokenize
import nltk
from tqdm import tqdm
import os

from utils import log_message, info, debug, warning, error

nltk.download('punkt', quiet=True)

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
        info("开始计算TF-IDF模型...")
        self.tfidf_vectorizer = TfidfVectorizer(
            tokenizer=DocumentProcessor.tokenize, 
            stop_words='english'
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.corpus)
        info("TF-IDF模型计算完成")
        return self.tfidf_vectorizer, self.tfidf_matrix
    
    def calculate_bm25(self):
        """计算BM25模型"""
        info("开始计算BM25模型...")
        
        # 使用tqdm包装分词过程
        tokenized_corpus = []
        for doc in tqdm(self.corpus, desc="分词处理中", unit="doc"):
            tokenized_corpus.append(DocumentProcessor.tokenize(doc))
        
        info("构建BM25索引中...")
        self.bm25 = BM25Okapi(tokenized_corpus)
        info("BM25索引构建完成")
        return self.bm25
    
    def save_models(self, output_prefix):
        """保存模型到文件"""
        if self.tfidf_vectorizer:
            info(f"正在保存TF-IDF模型到 {output_prefix}_tfidf.pkl...")
            with open(f'{output_prefix}_tfidf.pkl', 'wb') as f:
                pickle.dump(self.tfidf_vectorizer, f)
            info(f"TF-IDF模型已保存")
        if self.bm25:
            info(f"正在保存BM25模型到 {output_prefix}_bm25.pkl...")
            with open(f'{output_prefix}_bm25.pkl', 'wb') as f:
                pickle.dump(self.bm25, f)
            info(f"BM25模型已保存")


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
        
        info(f"开始处理文档，共 {total_lines} 行...")
        
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
                    error(f"解析JSON时出错，行内容: {line[:100]}... 错误: {e}")
                except KeyError as e:
                    error(f"文档中缺少关键字段: {e}")
    
    def save_processed_data(self):
        """保存处理后的数据"""
        plain_output = os.path.join(self.data_output_dir, "processed_plain.jsonl")
        markdown_output = os.path.join(self.data_output_dir, "processed_markdown.jsonl")
        
        info(f"正在保存处理后的纯文本数据到 {plain_output}...")
        with open(plain_output, 'w', encoding='utf-8') as f:
            # Using tqdm to show progress bar
            for item in tqdm(self.processed_plain, desc="保存纯文本数据", unit="doc"):
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        info(f"正在保存处理后的markdown数据到 {markdown_output}...")
        with open(markdown_output, 'w', encoding='utf-8') as f:
            # Using tqdm to show progress bar
            for item in tqdm(self.processed_markdown, desc="保存markdown数据", unit="doc"):
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        info(f"纯文本数据已保存到 {plain_output}")
        info(f"Markdown数据已保存到 {markdown_output}")
    
    def build_models(self):
        """构建搜索模型"""
        plain_corpus = [doc['text'] for doc in self.processed_plain]
        markdown_corpus = [doc['text'] for doc in self.processed_markdown]
        
        info("开始为纯文本构建模型...")
        plain_indexer = DocumentIndexer(plain_corpus)
        plain_indexer.calculate_tfidf()
        plain_indexer.calculate_bm25()
        plain_indexer.save_models(os.path.join(self.model_output_dir, "plain"))
        
        info("开始为markdown构建模型...")
        markdown_indexer = DocumentIndexer(markdown_corpus)
        markdown_indexer.calculate_tfidf()
        markdown_indexer.calculate_bm25()
        markdown_indexer.save_models(os.path.join(self.model_output_dir, "markdown"))
        
        info("所有模型已成功构建并保存")
    
    def run(self):
        """运行整个处理流程"""
        self.process_documents()
        self.save_processed_data()
        self.build_models()

'''
if __name__ == "__main__":
    # 配置参数
    input_file = "/Users/cusnsheep/Documents/dsa_course/S2/COMP5423/project/data_and_code(1)/data/documents.jsonl"
    data_output_dir = "/Users/cusnsheep/Documents/dsa_course/S2/COMP5423/project/data_and_code(1)/data/output/"
    model_output_dir = "/Users/cusnsheep/Documents/dsa_course/S2/COMP5423/project/data_and_code(1)/data/model/"
    
    # 创建并运行管道
    pipeline = DocumentPipeline(input_file, model_output_dir, data_output_dir)
    pipeline.run()
'''
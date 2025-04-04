import json
import re
from bs4 import BeautifulSoup
import html2text
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import pickle
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

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
        self.tfidf_vectorizer = TfidfVectorizer(
            tokenizer=DocumentProcessor.tokenize, 
            stop_words='english'
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.corpus)
        return self.tfidf_vectorizer, self.tfidf_matrix
    
    def calculate_bm25(self):
        """计算BM25模型"""
        tokenized_corpus = [DocumentProcessor.tokenize(doc) for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        return self.bm25
    
    def save_models(self, output_prefix):
        """保存模型到文件"""
        if self.tfidf_vectorizer:
            with open(f'{output_prefix}_tfidf.pkl', 'wb') as f:
                pickle.dump(self.tfidf_vectorizer, f)
        if self.bm25:
            with open(f'{output_prefix}_bm25.pkl', 'wb') as f:
                pickle.dump(self.bm25, f)


class DocumentPipeline:
    """文档处理管道，整合整个流程"""
    
    def __init__(self, input_path, output_dir):
        self.input_path = input_path
        self.output_dir = output_dir
        self.processed_plain = []
        self.processed_markdown = []
    
    def process_documents(self):
        """处理原始数据"""
        with open(self.input_path, 'r', encoding='utf-8') as f:
            for line in f:
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
                    print(f"Error decoding JSON for line: {line[:100]}... Error: {e}")
                except KeyError as e:
                    print(f"Missing key in document: {e}")
    
    def save_processed_data(self):
        """保存处理后的数据"""
        plain_output = f"{self.output_dir}processed_plain.jsonl"
        markdown_output = f"{self.output_dir}processed_markdown.jsonl"
        
        with open(plain_output, 'w', encoding='utf-8') as f:
            for item in self.processed_plain:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        with open(markdown_output, 'w', encoding='utf-8') as f:
            for item in self.processed_markdown:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"Saved plain text data to {plain_output}")
        print(f"Saved markdown data to {markdown_output}")
    
    def build_models(self):
        """构建搜索模型"""
        plain_corpus = [doc['text'] for doc in self.processed_plain]
        markdown_corpus = [doc['text'] for doc in self.processed_markdown]
        
        print("Building models for plain text...")
        plain_indexer = DocumentIndexer(plain_corpus)
        plain_indexer.calculate_tfidf()
        plain_indexer.calculate_bm25()
        plain_indexer.save_models(f"{self.output_dir}plain")
        
        print("Building models for markdown...")
        markdown_indexer = DocumentIndexer(markdown_corpus)
        markdown_indexer.calculate_tfidf()
        markdown_indexer.calculate_bm25()
        markdown_indexer.save_models(f"{self.output_dir}markdown")
        
        print("Models built and saved successfully")
    
    def run(self):
        """运行整个处理流程"""
        print("Processing documents...")
        self.process_documents()
        self.save_processed_data()
        self.build_models()


if __name__ == "__main__":
    # 配置参数
    input_file = "/Users/cusnsheep/Documents/dsa_course/S2/COMP5423/project/data_and_code(1)/data/documents.jsonl"
    output_dir = "/Users/cusnsheep/Documents/dsa_course/S2/COMP5423/project/data_and_code(1)/data/output/"
    
    # 创建并运行管道
    pipeline = DocumentPipeline(input_file, output_dir)
    pipeline.run()
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
                
    ### 增加load model      
     def load_models(self, tfidf_path, bm25_path):
        """从文件加载模型"""
        with open(tfidf_path, 'rb') as f:
            self.tfidf_vectorizer = pickle.load(f)
        with open(bm25_path, 'rb') as f:
            self.bm25 = pickle.load(f)
        print("Models loaded from files successfully")



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
        
    ### 增加加载模型
    def load_models(self):
        """加载模型"""
        self.plain_indexer = DocumentIndexer([])
        self.markdown_indexer = DocumentIndexer([])
        
        self.plain_indexer.load_models(f'{self.output_dir}plain_tfidf.pkl', f'{self.output_dir}plain_bm25.pkl')
        self.markdown_indexer.load_models(f'{self.output_dir}markdown_tfidf.pkl', f'{self.output_dir}markdown_bm25.pkl')
        
        print("Models loaded successfully")
    
    def hybrid_search(self, query, top_n=5):
        """进行混合搜索"""
        # TF-IDF search
        tfidf_vec_plain = self.plain_indexer.tfidf_vectorizer.transform([query])
        tfidf_scores_plain = self.plain_indexer.tfidf_matrix.dot(tfidf_vec_plain.T).toarray().flatten()
        
        tfidf_vec_markdown = self.markdown_indexer.tfidf_vectorizer.transform([query])
        tfidf_scores_markdown = self.markdown_indexer.tfidf_matrix.dot(tfidf_vec_markdown.T).toarray().flatten()
        
        # BM25 search
        bm25_scores_plain = self.plain_indexer.bm25.get_scores(DocumentProcessor.tokenize(query))
        bm25_scores_markdown = self.markdown_indexer.bm25.get_scores(DocumentProcessor.tokenize(query))
        
        # Combine scores
        combined_scores_plain = tfidf_scores_plain + bm25_scores_plain
        combined_scores_markdown = tfidf_scores_markdown + bm25_scores_markdown
        
        # Get top_n results
        top_plain_idx = combined_scores_plain.argsort()[-top_n:][::-1]
        top_markdown_idx = combined_scores_markdown.argsort()[-top_n:][::-1]
        
        results_plain = [(self.processed_plain[i]['doc_id'], combined_scores_plain[i]) for i in top_plain_idx]
        results_markdown = [(self.processed_markdown[i]['doc_id'], combined_scores_markdown[i]) for i in top_markdown_idx]
        
        return results_plain, results_markdown
    
    def run(self):
        """运行整个处理流程"""
        print("Processing documents...")
        self.process_documents()
        self.save_processed_data()
        self.build_models() 
        self.load_models()# 训练好的模型，我要怎么调用这个模型，进行hybrid search呢？
        # 还需要提供一个接口，我们可以输入query,返回document id, score, rank之类
        # 另外，需要在val data跑一遍tf-idf和bm25的模型，返回结果，按照老师给的格式，保存到文件中
        # hybrid search query
        query = "example search query"
        results_plain, results_markdown = self.hybrid_search(query)
        print("Plain Text Results:", results_plain)
        print("Markdown Results:", results_markdown)

if __name__ == "__main__":
    # 配置参数
    input_file = "/Users/cusnsheep/Documents/dsa_course/S2/COMP5423/project/data_and_code(1)/data/documents.jsonl"
    output_dir = "/Users/cusnsheep/Documents/dsa_course/S2/COMP5423/project/data_and_code(1)/data/output/"
    
    # 创建并运行管道
    pipeline = DocumentPipeline(input_file, output_dir)
    pipeline.run()

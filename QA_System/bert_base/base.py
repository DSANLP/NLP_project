import os
import pickle
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
import faiss
from utils import log_message, info, debug, warning, error

class DensePassageRetriever:
    """
    DPR检索系统基础类
    
    功能：
    - 使用预训练的BERT模型进行文档和问题的向量化
    - 使用FAISS进行高效向量检索
    """
    
    def __init__(self, config=None):
        """
        初始化DPR检索系统
        
        参数:
            config: 配置对象，包含模型和检索参数
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.get('system', {}).get('device') == 'cuda' else 'cpu')
        self.model_path = config.get('retrieval', {}).get('deep_retrieval', {}).get('model', 'bert-base-chinese')
        self.tokenizer = None
        self.question_encoder = None
        self.context_encoder = None
        self.doc_embeddings = None
        self.faiss_index = None
        self.doc_ids = None

    def load_model(self, model_dir=None):
        """
        加载预训练或微调过的DPR模型
        
        参数:
            model_dir: 模型保存目录，如果为None则使用默认配置路径
            
        返回:
            加载模型是否成功
        """
        if model_dir is None:
            model_dir = os.path.join(self.config.get('paths', {}).get('model_root', './models/'), 'dpr')
        
        try:
            # 加载预训练tokenizer
            self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
            
            # 加载问题编码器
            question_encoder_path = os.path.join(model_dir, 'question_encoder')
            if os.path.exists(question_encoder_path):
                self.question_encoder = BertModel.from_pretrained(question_encoder_path)
                debug(f"从{question_encoder_path}加载问题编码器")
            else:
                self.question_encoder = BertModel.from_pretrained(self.model_path)
                debug(f"使用预训练模型{self.model_path}作为问题编码器")
            
            # 加载文档编码器
            context_encoder_path = os.path.join(model_dir, 'context_encoder')
            if os.path.exists(context_encoder_path):
                self.context_encoder = BertModel.from_pretrained(context_encoder_path)
                debug(f"从{context_encoder_path}加载文档编码器")
            else:
                self.context_encoder = BertModel.from_pretrained(self.model_path)
                debug(f"使用预训练模型{self.model_path}作为文档编码器")
            
            # 将模型移动到对应设备
            self.question_encoder = self.question_encoder.to(self.device)
            self.context_encoder = self.context_encoder.to(self.device)
            
            # 设置模型为评估模式
            self.question_encoder.eval()
            self.context_encoder.eval()
            
            return True
        except Exception as e:
            error(f"加载DPR模型失败: {str(e)}")
            return False

    def encode_questions(self, questions, batch_size=16):
        """
        将问题列表编码为向量表示
        
        参数:
            questions: 问题文本列表
            batch_size: 批处理大小
            
        返回:
            问题的向量表示 (numpy数组)
        """
        encodings = []
        
        # 分批处理
        for i in tqdm(range(0, len(questions), batch_size), desc="编码问题"):
            batch = questions[i:i+batch_size]
            inputs = self.tokenizer(
                batch, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.question_encoder(**inputs)
                # 使用[CLS]标记的表示作为问题向量
                batch_encodings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                encodings.append(batch_encodings)
                
        return np.vstack(encodings)

    def encode_contexts(self, contexts, batch_size=8):
        """
        将文档上下文列表编码为向量表示
        
        参数:
            contexts: 文档文本列表
            batch_size: 批处理大小
            
        返回:
            文档的向量表示 (numpy数组)
        """
        encodings = []
        
        # 分批处理
        for i in tqdm(range(0, len(contexts), batch_size), desc="编码文档"):
            batch = contexts[i:i+batch_size]
            inputs = self.tokenizer(
                batch, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.context_encoder(**inputs)
                # 使用[CLS]标记的表示作为文档向量
                batch_encodings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                encodings.append(batch_encodings)
                
        return np.vstack(encodings)

    def build_faiss_index(self, doc_embeddings, index_type="Flat"):
        """
        使用FAISS构建向量索引
        
        参数:
            doc_embeddings: 文档向量表示
            index_type: FAISS索引类型 ('Flat', 'IVF', 'HNSW')
            
        返回:
            构建的FAISS索引
        """
        vector_dim = doc_embeddings.shape[1]
        
        if index_type == "Flat":
            # 最简单的精确检索索引
            index = faiss.IndexFlatIP(vector_dim)  # 使用内积距离 (归一化向量后等价于余弦相似度)
        elif index_type == "IVF":
            # IVF索引，适用于大规模数据
            nlist = min(4096, max(10, int(doc_embeddings.shape[0] / 39)))  # 选择聚类中心数量
            quantizer = faiss.IndexFlatIP(vector_dim)
            index = faiss.IndexIVFFlat(quantizer, vector_dim, nlist, faiss.METRIC_INNER_PRODUCT)
            # 需要训练，用数据训练
            index.train(doc_embeddings)
        elif index_type == "HNSW":
            # HNSW索引，平衡速度和准确性
            index = faiss.IndexHNSWFlat(vector_dim, 32, faiss.METRIC_INNER_PRODUCT)
        else:
            raise ValueError(f"不支持的索引类型: {index_type}")
        
        # 归一化向量，以便使用内积计算余弦相似度
        faiss.normalize_L2(doc_embeddings)
        
        # 将文档向量添加到索引
        index.add(doc_embeddings)
        
        return index

    def save_index(self, output_dir, overwrite=False):
        """
        保存FAISS索引和相关数据
        
        参数:
            output_dir: 输出目录
            overwrite: 是否覆盖现有文件
            
        返回:
            保存是否成功
        """
        if self.faiss_index is None or self.doc_ids is None:
            error("没有索引可以保存，请先构建索引")
            return False
            
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存FAISS索引
            index_path = os.path.join(output_dir, "faiss_index.bin")
            if os.path.exists(index_path) and not overwrite:
                warning(f"文件{index_path}已存在且不覆盖，跳过保存")
            else:
                faiss.write_index(self.faiss_index, index_path)
                info(f"FAISS索引已保存到{index_path}")
            
            # 保存文档ID映射
            doc_ids_path = os.path.join(output_dir, "doc_ids.pkl")
            if os.path.exists(doc_ids_path) and not overwrite:
                warning(f"文件{doc_ids_path}已存在且不覆盖，跳过保存")
            else:
                with open(doc_ids_path, 'wb') as f:
                    pickle.dump(self.doc_ids, f)
                info(f"文档ID映射已保存到{doc_ids_path}")
                
            return True
        except Exception as e:
            error(f"保存索引失败: {str(e)}")
            return False

    def load_index(self, input_dir):
        """
        加载FAISS索引和相关数据
        
        参数:
            input_dir: 输入目录
            
        返回:
            加载是否成功
        """
        try:
            # 加载FAISS索引
            index_path = os.path.join(input_dir, "faiss_index.bin")
            if not os.path.exists(index_path):
                error(f"FAISS索引文件不存在: {index_path}")
                return False
                
            self.faiss_index = faiss.read_index(index_path)
            info(f"已加载FAISS索引，包含{self.faiss_index.ntotal}个向量")
            
            # 加载文档ID映射
            doc_ids_path = os.path.join(input_dir, "doc_ids.pkl")
            if not os.path.exists(doc_ids_path):
                error(f"文档ID映射文件不存在: {doc_ids_path}")
                return False
                
            with open(doc_ids_path, 'rb') as f:
                self.doc_ids = pickle.load(f)
            info(f"已加载文档ID映射，包含{len(self.doc_ids)}个文档ID")
            
            return True
        except Exception as e:
            error(f"加载索引失败: {str(e)}")
            return False

    def search(self, query, top_k=5):
        """
        搜索与查询最相关的文档
        
        参数:
            query: 查询文本
            top_k: 返回的最相关文档数量
            
        返回:
            包含文档ID和相关性得分的字典
        """
        if self.faiss_index is None or self.doc_ids is None:
            error("索引未加载，无法执行搜索")
            return {"document_id": [], "scores": []}
            
        # 编码查询
        query_vector = self.encode_questions([query])[0].reshape(1, -1)
        
        # 归一化查询向量
        faiss.normalize_L2(query_vector)
        
        # 执行搜索
        scores, indices = self.faiss_index.search(query_vector, k=top_k)
        
        # 获取文档ID
        retrieved_doc_ids = [self.doc_ids[idx] for idx in indices[0] if idx < len(self.doc_ids)]
        retrieved_scores = scores[0].tolist()
        
        return {
            "document_id": retrieved_doc_ids,
            "scores": retrieved_scores
        }

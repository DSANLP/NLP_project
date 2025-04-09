import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, AutoTokenizer
from typing import Dict, List, Tuple, Optional, Union

class DPREncoder(nn.Module):
    """
    DPR编码器模块，基于BERT
    """
    def __init__(
        self,
        model_name: str = "nomic-ai/nomic-bert-2048",
        pooling: str = "cls",
        normalize: bool = True
    ):
        """
        初始化编码器
        
        Args:
            model_name: 使用的预训练模型
            pooling: 池化方式 ('cls', 'mean', 'max')
            normalize: 是否对输出向量进行L2归一化
        """
        super(DPREncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.pooling = pooling
        self.normalize = normalize
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input_ids: 输入token ID
            attention_mask: 注意力掩码
            token_type_ids: token类型ID
            
        Returns:
            编码后的向量
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # 获取最后一层的隐藏状态
        hidden_state = outputs[0]  # [batch_size, seq_len, hidden_size]
        
        # 根据池化方式选择不同的处理方法
        if self.pooling == "cls":
            # 使用[CLS]标记的表示作为序列表示
            vector = hidden_state[:, 0, :]  # [batch_size, hidden_size]
        elif self.pooling == "mean":
            # 对整个序列取平均
            vector = torch.mean(hidden_state, dim=1)  # [batch_size, hidden_size]
        elif self.pooling == "max":
            # 对整个序列取最大值
            vector = torch.max(hidden_state, dim=1)[0]  # [batch_size, hidden_size]
        else:
            raise ValueError(f"不支持的池化方式: {self.pooling}")
        
        # 归一化向量
        if self.normalize:
            vector = F.normalize(vector, p=2, dim=1)
            
        return vector


class DPRModel(nn.Module):
    """
    DPR双塔模型
    """
    def __init__(
        self,
        query_encoder_name: str = "nomic-ai/nomic-bert-2048",
        ctx_encoder_name: str = "nomic-ai/nomic-bert-2048",
        shared_weights: bool = False,
        temperature: float = 0.05
    ):
        """
        初始化DPR模型
        
        Args:
            query_encoder_name: 查询编码器使用的预训练模型
            ctx_encoder_name: 文本编码器使用的预训练模型 
            shared_weights: 是否共享两个编码器的权重
            temperature: 相似度计算的温度系数
        """
        super(DPRModel, self).__init__()
        
        # 创建问题编码器
        self.query_encoder = DPREncoder(model_name=query_encoder_name)
        
        # 是否共享权重
        if shared_weights:
            self.ctx_encoder = self.query_encoder
        else:
            self.ctx_encoder = DPREncoder(model_name=ctx_encoder_name)
            
        self.temperature = temperature
    
    def encode_query(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """编码查询"""
        return self.query_encoder(input_ids, attention_mask, token_type_ids)
    
    def encode_ctx(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """编码文本上下文"""
        return self.ctx_encoder(input_ids, attention_mask, token_type_ids)
    
    def forward(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        query_token_type_ids: Optional[torch.Tensor] = None,
        pos_ctx_input_ids: Optional[torch.Tensor] = None,
        pos_ctx_attention_mask: Optional[torch.Tensor] = None,
        pos_ctx_token_type_ids: Optional[torch.Tensor] = None,
        neg_ctx_input_ids: Optional[torch.Tensor] = None,
        neg_ctx_attention_mask: Optional[torch.Tensor] = None,
        neg_ctx_token_type_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            query_*: 查询文本的输入
            pos_ctx_*: 正样本文本的输入
            neg_ctx_*: 负样本文本的输入
            
        Returns:
            (query_vectors, pos_ctx_vectors, neg_ctx_vectors)
        """
        # 编码查询
        query_vectors = self.encode_query(
            query_input_ids,
            query_attention_mask,
            query_token_type_ids
        )
        
        # 编码正样本文本
        pos_ctx_vectors = None
        if pos_ctx_input_ids is not None:
            pos_ctx_vectors = self.encode_ctx(
                pos_ctx_input_ids,
                pos_ctx_attention_mask,
                pos_ctx_token_type_ids
            )
        
        # 编码负样本文本
        neg_ctx_vectors = None
        if neg_ctx_input_ids is not None:
            # 检查负样本输入是否为三维张量 [batch_size, n_neg, seq_len]
            if len(neg_ctx_input_ids.shape) == 3:
                batch_size, n_neg, seq_len = neg_ctx_input_ids.shape
                
                # 重塑为二维张量 [batch_size * n_neg, seq_len]
                neg_ctx_input_ids = neg_ctx_input_ids.reshape(batch_size * n_neg, seq_len)
                neg_ctx_attention_mask = neg_ctx_attention_mask.reshape(batch_size * n_neg, seq_len)
                
                if neg_ctx_token_type_ids is not None:
                    neg_ctx_token_type_ids = neg_ctx_token_type_ids.reshape(batch_size * n_neg, seq_len)
                
                # 编码负样本
                neg_ctx_vectors = self.encode_ctx(
                    neg_ctx_input_ids,
                    neg_ctx_attention_mask,
                    neg_ctx_token_type_ids
                )
                
                # 重塑回三维张量 [batch_size, n_neg, hidden_size]
                hidden_size = neg_ctx_vectors.shape[-1]  # 使用最后一个维度作为hidden_size
                neg_ctx_vectors = neg_ctx_vectors.reshape(batch_size, n_neg, hidden_size)
            else:
                # 正常编码
                neg_ctx_vectors = self.encode_ctx(
                    neg_ctx_input_ids,
                    neg_ctx_attention_mask,
                    neg_ctx_token_type_ids
                )
        
        return query_vectors, pos_ctx_vectors, neg_ctx_vectors
    
    def compute_similarity(
        self,
        query_vectors: torch.Tensor,
        ctx_vectors: torch.Tensor
    ) -> torch.Tensor:
        """
        计算查询和上下文向量之间的相似度
        
        Args:
            query_vectors: 查询向量 [batch_size, hidden_size]
            ctx_vectors: 上下文向量 [batch_size, hidden_size] 或 [batch_size, n_contexts, hidden_size]
            
        Returns:
            相似度分数 [batch_size] 或 [batch_size, n_contexts]
        """
        # 检查ctx_vectors的维度
        if len(ctx_vectors.shape) == 3:
            # [batch_size, n_contexts, hidden_size]
            batch_size, n_contexts, hidden_size = ctx_vectors.shape
            # 重塑query_vectors以便广播
            query_vectors = query_vectors.unsqueeze(1).expand(-1, n_contexts, -1)
            # 计算点积
            scores = torch.sum(query_vectors * ctx_vectors, dim=2) / self.temperature
        else:
            # [batch_size, hidden_size]
            scores = torch.sum(query_vectors * ctx_vectors, dim=1) / self.temperature
            
        return scores
        
    def compute_loss(
        self,
        query_vectors: torch.Tensor,
        pos_ctx_vectors: torch.Tensor,
        neg_ctx_vectors: torch.Tensor
    ) -> torch.Tensor:
        """
        计算对比损失
        
        Args:
            query_vectors: 查询向量 [batch_size, hidden_size]
            pos_ctx_vectors: 正样本向量 [batch_size, hidden_size]
            neg_ctx_vectors: 负样本向量 [batch_size, n_neg, hidden_size]
            
        Returns:
            对比损失
        """
        # 确保neg_ctx_vectors是三维张量
        if len(neg_ctx_vectors.shape) != 3:
            raise ValueError(f"neg_ctx_vectors应该是三维张量，但得到了形状: {neg_ctx_vectors.shape}")
        
        batch_size, n_neg, _ = neg_ctx_vectors.shape
        
        # 计算与正样本的相似度
        pos_scores = self.compute_similarity(query_vectors, pos_ctx_vectors)  # [batch_size]
        
        # 计算与负样本的相似度
        neg_scores = self.compute_similarity(query_vectors, neg_ctx_vectors)  # [batch_size, n_neg]
        
        # 创建得分矩阵 [batch_size, 1 + n_neg]
        scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
        
        # 目标是使正样本得分最高，即索引0
        target = torch.zeros(batch_size, dtype=torch.long, device=scores.device)
        
        # 使用交叉熵损失
        loss = F.cross_entropy(scores, target)
        
        return loss
        
    def save_encoders(self, output_dir: str) -> None:
        """
        保存编码器模型
        
        Args:
            output_dir: 输出目录
        """
        import os
        
        # 保存查询编码器
        query_output_dir = os.path.join(output_dir, "query_encoder")
        os.makedirs(query_output_dir, exist_ok=True)
        self.query_encoder.bert.save_pretrained(query_output_dir)
        
        # 如果不共享权重，保存上下文编码器
        if self.query_encoder != self.ctx_encoder:
            ctx_output_dir = os.path.join(output_dir, "ctx_encoder")
            os.makedirs(ctx_output_dir, exist_ok=True)
            self.ctx_encoder.bert.save_pretrained(ctx_output_dir)
            
    @classmethod
    def load_encoders(
        cls,
        model_dir: str,
        shared_weights: bool = False,
        temperature: float = 0.05
    ) -> "DPRModel":
        """
        从保存的目录加载编码器模型
        
        Args:
            model_dir: 模型目录
            shared_weights: 是否共享权重
            temperature: 温度系数
            
        Returns:
            加载好的DPR模型
        """
        import os
        
        query_encoder_path = os.path.join(model_dir, "query_encoder")
        ctx_encoder_path = os.path.join(model_dir, "ctx_encoder")
        
        if shared_weights or not os.path.exists(ctx_encoder_path):
            return cls(
                query_encoder_name=query_encoder_path,
                ctx_encoder_name=query_encoder_path,
                shared_weights=True,
                temperature=temperature
            )
        else:
            return cls(
                query_encoder_name=query_encoder_path,
                ctx_encoder_name=ctx_encoder_path,
                shared_weights=False,
                temperature=temperature
            )
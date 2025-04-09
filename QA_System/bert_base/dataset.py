import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union
from transformers import AutoTokenizer

class DPRDataset(Dataset):
    """
    DPR模型的数据集
    """
    def __init__(
        self,
        data_file: str,
        tokenizer: AutoTokenizer,
        max_query_length: int = 128,
        max_ctx_length: int = 2048,
        is_training: bool = True
    ):
        """
        初始化数据集
        
        Args:
            data_file: 数据文件路径
            tokenizer: 分词器
            max_query_length: 查询文本的最大长度
            max_ctx_length: 上下文文本的最大长度 (利用nomic-bert-2048的长序列能力)
            is_training: 是否为训练模式
        """
        self.tokenizer = tokenizer
        self.max_query_length = max_query_length
        self.max_ctx_length = max_ctx_length
        self.is_training = is_training
        
        # 尝试两种方式加载数据
        try:
            # 首先尝试作为普通JSON加载
            with open(data_file, 'r', encoding='utf-8') as f:
                json_obj = json.load(f)
                
            # 如果是单个JSON对象，将其转换为列表
            if isinstance(json_obj, dict):
                self.data = [json_obj]
            else:
                self.data = json_obj
        except json.JSONDecodeError:
            # 如果失败，尝试按行加载JSONL格式
            self.data = []
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            item = json.loads(line)
                            self.data.append(item)
                        except json.JSONDecodeError:
                            continue
            
        print(f"加载了 {len(self.data)} 条数据")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取数据项
        
        Args:
            idx: 数据索引
            
        Returns:
            包含模型输入的字典
        """
        item = self.data[idx]
        
        # 获取查询、正样本和负样本文本
        query = item["question"]
        
        # 兼容两种格式：positive_ctx（temp版本）和 positive_ctxs（原版本）
        if "positive_ctx" in item:
            # temp版本格式
            positive_ctx = item["positive_ctx"]
        elif "positive_ctxs" in item and isinstance(item["positive_ctxs"], list) and len(item["positive_ctxs"]) > 0:
            # 原始格式，但取出第一个正样本的文本
            if isinstance(item["positive_ctxs"][0], dict) and "text" in item["positive_ctxs"][0]:
                positive_ctx = item["positive_ctxs"][0]["text"]
            else:
                positive_ctx = item["positive_ctxs"][0]
        else:
            # 兜底，使用空字符串
            positive_ctx = ""
            
        # 获取负样本，兼容两种格式
        negative_ctxs = []
        if "negative_ctxs" in item:
            if isinstance(item["negative_ctxs"], list):
                if len(item["negative_ctxs"]) > 0:
                    if isinstance(item["negative_ctxs"][0], dict) and "text" in item["negative_ctxs"][0]:
                        # 原始格式，需要提取文本
                        negative_ctxs = [neg["text"] for neg in item["negative_ctxs"]]
                    else:
                        # temp版本，直接是文本列表
                        negative_ctxs = item["negative_ctxs"]
        
        # 对查询文本进行编码
        query_encoding = self.tokenizer(
            query,
            max_length=self.max_query_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 对正样本文本进行编码
        pos_ctx_encoding = self.tokenizer(
            positive_ctx,
            max_length=self.max_ctx_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 移除批次维度（DataLoader会添加批次维度）
        query_encoding = {k: v.squeeze(0) for k, v in query_encoding.items()}
        pos_ctx_encoding = {k: v.squeeze(0) for k, v in pos_ctx_encoding.items()}
        
        # 构建输入字典
        inputs = {
            "query_input_ids": query_encoding["input_ids"],
            "query_attention_mask": query_encoding["attention_mask"],
            "query_token_type_ids": query_encoding.get("token_type_ids", None),  # 兼容不返回token_type_ids的tokenizer
            "pos_ctx_input_ids": pos_ctx_encoding["input_ids"],
            "pos_ctx_attention_mask": pos_ctx_encoding["attention_mask"],
            "pos_ctx_token_type_ids": pos_ctx_encoding.get("token_type_ids", None),
        }
        
        # 如果是训练模式，添加负样本
        if self.is_training:
            # 对所有负样本文本进行编码
            neg_ctx_encodings = []
            for neg_ctx in negative_ctxs:
                neg_encoding = self.tokenizer(
                    neg_ctx,
                    max_length=self.max_ctx_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                # 移除批次维度
                neg_encoding = {k: v.squeeze(0) for k, v in neg_encoding.items()}
                neg_ctx_encodings.append(neg_encoding)
            
            # 堆叠所有负样本编码
            n_neg = len(neg_ctx_encodings)
            if n_neg > 0:
                neg_input_ids = torch.stack([enc["input_ids"] for enc in neg_ctx_encodings])
                neg_attention_mask = torch.stack([enc["attention_mask"] for enc in neg_ctx_encodings])
                
                if "token_type_ids" in neg_ctx_encodings[0]:
                    neg_token_type_ids = torch.stack([enc["token_type_ids"] for enc in neg_ctx_encodings])
                    inputs["neg_ctx_token_type_ids"] = neg_token_type_ids
                
                inputs.update({
                    "neg_ctx_input_ids": neg_input_ids,
                    "neg_ctx_attention_mask": neg_attention_mask,
                })
        
        return inputs


def create_dataloader(
    data_file: str,
    tokenizer: AutoTokenizer,
    batch_size: int = 8,
    max_query_length: int = 128,
    max_ctx_length: int = 2048,
    is_training: bool = True,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """
    创建数据加载器
    
    Args:
        data_file: 数据文件路径
        tokenizer: 分词器
        batch_size: 批次大小
        max_query_length: 查询文本的最大长度
        max_ctx_length: 上下文文本的最大长度 (利用nomic-bert-2048的长序列能力)
        is_training: 是否为训练模式
        shuffle: 是否打乱数据
        num_workers: 数据加载的工作进程数量
        
    Returns:
        DataLoader对象
    """
    dataset = DPRDataset(
        data_file=data_file,
        tokenizer=tokenizer,
        max_query_length=max_query_length,
        max_ctx_length=max_ctx_length,
        is_training=is_training
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader
import os
import json
import yaml
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import faiss
from sklearn.model_selection import train_test_split
import logging
import time

from bert_base.base import DensePassageRetriever
from utils import log_message, info, debug, warning, error

class DPRDataset(Dataset):
    """DPR训练数据集类"""
    
    def __init__(self, questions, contexts, tokenizer, max_length=512):
        """
        初始化数据集
        
        参数:
            questions: 问题文本列表
            contexts: 对应的正文本和负文本列表
            tokenizer: 用于tokenize文本的tokenizer
            max_length: 序列最大长度
        """
        self.questions = questions
        self.contexts = contexts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = self.questions[idx]
        
        # 获取正文本和负文本
        # contexts[idx]应该是一个字典，包含'positive'和'negative'字段
        positive_context = self.contexts[idx]['positive']
        negative_contexts = self.contexts[idx]['negative']
        
        # tokenize问题
        question_encoding = self.tokenizer(
            question,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # tokenize正文本
        positive_encoding = self.tokenizer(
            positive_context,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # tokenize负文本
        negative_encodings = []
        for neg_context in negative_contexts:
            neg_encoding = self.tokenizer(
                neg_context,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt'
            )
            negative_encodings.append({
                'input_ids': neg_encoding['input_ids'].squeeze(0),
                'attention_mask': neg_encoding['attention_mask'].squeeze(0),
                'token_type_ids': neg_encoding['token_type_ids'].squeeze(0) if 'token_type_ids' in neg_encoding else None
            })
        
        # 返回张量字典
        return {
            'question_input_ids': question_encoding['input_ids'].squeeze(0),
            'question_attention_mask': question_encoding['attention_mask'].squeeze(0),
            'question_token_type_ids': question_encoding['token_type_ids'].squeeze(0) if 'token_type_ids' in question_encoding else None,
            'positive_input_ids': positive_encoding['input_ids'].squeeze(0),
            'positive_attention_mask': positive_encoding['attention_mask'].squeeze(0),
            'positive_token_type_ids': positive_encoding['token_type_ids'].squeeze(0) if 'token_type_ids' in positive_encoding else None,
            'negative_contexts': negative_encodings,
            'num_negatives': len(negative_contexts)
        }


class DPRTrainer:
    """DPR模型训练器"""
    
    def __init__(self, config_path="config.yaml"):
        """
        初始化训练器
        
        参数:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        info(f"[DPRTrainer] 初始化训练器，配置文件路径: {config_path}")
        
        start_time = time.time()
        self.config = self.load_config(config_path)
        self.dpr_config = self.config.get("retrieval", {}).get("deep_retrieval", {})
        
        # 设置随机种子
        random_seed = self.config.get("system", {}).get("seed", 42)
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
        
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() and 
                                  self.config.get("system", {}).get("device") == "cuda" else "cpu")
        
        # 设置模型路径
        self.model_path = self.dpr_config.get("model", "bert-base-chinese")
        
        # 设置训练参数
        self.train_params = {
            "batch_size": self.dpr_config.get("train_batch_size", 8),
            "epochs": self.dpr_config.get("epochs", 3),
            "learning_rate": self.dpr_config.get("learning_rate", 2e-5),
            "warmup_steps": self.dpr_config.get("warmup_steps", 0),
            "weight_decay": self.dpr_config.get("weight_decay", 0.01),
            "max_grad_norm": self.dpr_config.get("max_grad_norm", 1.0),
            "num_negatives": self.dpr_config.get("num_negatives", 7),
            "max_length": self.dpr_config.get("max_length", 512),
        }
        
        # 设置文件路径
        self.data_paths = {
            "train_path": os.path.join(self.config.get("data", {}).get("input_file", "./data/original_data/"), "train.jsonl"),
            "val_path": self.config.get("data", {}).get("val_path", "./data/original_data/val.jsonl"),
            "processed_data_dir": os.path.join(self.config.get("data", {}).get("data_output_dir", "./data/processed_data/")),
            "model_output_dir": os.path.join(self.config.get("paths", {}).get("model_root", "./models/"), "dpr")
        }
        
        # 确保路径以斜杠结尾
        for key in ["processed_data_dir", "model_output_dir"]:
            if not self.data_paths[key].endswith("/"):
                self.data_paths[key] += "/"
        
        # 转换为绝对路径
        for key in self.data_paths:
            self.data_paths[key] = os.path.abspath(self.data_paths[key])
        
        debug(f"[DPRTrainer] 配置加载完成，耗时: {time.time() - start_time:.2f}秒")
        info(f"[DPRTrainer] 训练参数: {', '.join([f'{k}={v}' for k, v in self.train_params.items()])}")
    
    def load_config(self, config_path):
        """加载配置文件"""
        debug(f"[DPRTrainer] 开始加载配置文件: {config_path}")
        if not os.path.exists(config_path):
            error_msg = f"配置文件不存在: {config_path}"
            error(f"[DPRTrainer] {error_msg}")
            raise FileNotFoundError(error_msg)
            
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            debug(f"[DPRTrainer] 配置文件加载成功")
            return config
        except Exception as e:
            error_msg = f"加载配置文件失败: {str(e)}"
            error(f"[DPRTrainer] {error_msg}")
            raise
    
    def load_raw_data(self):
        """加载原始训练和验证数据"""
        train_path = self.data_paths["train_path"]
        val_path = self.data_paths["val_path"]
        
        train_data = []
        val_data = []
        
        # 加载训练数据
        info(f"[DPRTrainer] 加载训练数据: {train_path}")
        with open(train_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    train_data.append(data)
                except json.JSONDecodeError:
                    warning(f"[DPRTrainer] 无法解析训练数据行: {line[:50]}...")
        
        # 加载验证数据
        info(f"[DPRTrainer] 加载验证数据: {val_path}")
        with open(val_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    val_data.append(data)
                except json.JSONDecodeError:
                    warning(f"[DPRTrainer] 无法解析验证数据行: {line[:50]}...")
        
        info(f"[DPRTrainer] 已加载{len(train_data)}条训练数据和{len(val_data)}条验证数据")
        return train_data, val_data
    
    def load_processed_documents(self):
        """加载处理后的文档数据"""
        use_plain_text = self.config.get("retrieval", {}).get("use_plain_text", True)
        doc_type = "plain" if use_plain_text else "markdown"
        doc_path = os.path.join(self.data_paths["processed_data_dir"], f"processed_{doc_type}.jsonl")
        
        info(f"[DPRTrainer] 加载处理后的文档: {doc_path}")
        
        documents = {}
        try:
            with open(doc_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        doc = json.loads(line)
                        documents[doc["doc_id"]] = doc["text"]
                    except json.JSONDecodeError:
                        warning(f"[DPRTrainer] 无法解析文档数据行: {line[:50]}...")
            
            info(f"[DPRTrainer] 已加载{len(documents)}个处理后的文档")
            return documents
        except FileNotFoundError:
            error(f"[DPRTrainer] 文档文件不存在: {doc_path}")
            error(f"[DPRTrainer] 请先运行处理模式(process_mode)生成必要的文档文件")
            raise
    
    def prepare_training_data(self, train_data, val_data, documents):
        """准备训练和验证数据集"""
        num_negatives = self.train_params["num_negatives"]
        
        # 初始化数据结构
        train_questions = []
        train_contexts = []
        val_questions = []
        val_contexts = []
        
        # 获取所有文档ID，用于采样负例
        all_doc_ids = list(documents.keys())
        
        # 处理训练数据
        info(f"[DPRTrainer] 准备训练数据集，每个问题使用{num_negatives}个负例...")
        for item in tqdm(train_data, desc="处理训练数据"):
            question = item.get("query", "")
            positive_doc_id = item.get("document_id", "")
            
            # 忽略缺少问题或正例文档ID的样本
            if not question or not positive_doc_id:
                continue
            
            # 获取正例文档文本
            positive_context = documents.get(positive_doc_id, "")
            if not positive_context:
                continue  # 忽略找不到文档的样本
            
            # 采样负例文档ID，排除正例
            negative_doc_ids = random.sample([doc_id for doc_id in all_doc_ids if doc_id != positive_doc_id], 
                                           min(num_negatives, len(all_doc_ids) - 1))
            
            # 获取负例文档文本
            negative_contexts = [documents.get(doc_id, "") for doc_id in negative_doc_ids]
            negative_contexts = [ctx for ctx in negative_contexts if ctx]  # 过滤空文本
            
            # 如果找到了足够的负例，则添加到数据集
            if negative_contexts:
                train_questions.append(question)
                train_contexts.append({
                    "positive": positive_context,
                    "negative": negative_contexts
                })
        
        # 处理验证数据
        info(f"[DPRTrainer] 准备验证数据集，每个问题使用{num_negatives}个负例...")
        for item in tqdm(val_data, desc="处理验证数据"):
            question = item.get("query", "")
            positive_doc_id = item.get("document_id", "")
            
            # 忽略缺少问题或正例文档ID的样本
            if not question or not positive_doc_id:
                continue
            
            # 获取正例文档文本
            positive_context = documents.get(positive_doc_id, "")
            if not positive_context:
                continue  # 忽略找不到文档的样本
            
            # 采样负例文档ID，排除正例
            negative_doc_ids = random.sample([doc_id for doc_id in all_doc_ids if doc_id != positive_doc_id], 
                                           min(num_negatives, len(all_doc_ids) - 1))
            
            # 获取负例文档文本
            negative_contexts = [documents.get(doc_id, "") for doc_id in negative_doc_ids]
            negative_contexts = [ctx for ctx in negative_contexts if ctx]  # 过滤空文本
            
            # 如果找到了足够的负例，则添加到数据集
            if negative_contexts:
                val_questions.append(question)
                val_contexts.append({
                    "positive": positive_context,
                    "negative": negative_contexts
                })
        
        info(f"[DPRTrainer] 准备了{len(train_questions)}条训练样本和{len(val_questions)}条验证样本")
        return train_questions, train_contexts, val_questions, val_contexts
    
    def evaluate_model(self, model, tokenizer, eval_questions, eval_doc_ids, documents):
        """
        评估模型在验证集上的性能
        
        参数:
            model: 模型（问题编码器）
            tokenizer: 分词器
            eval_questions: 评估问题列表
            eval_doc_ids: 评估问题对应的正确文档ID列表
            documents: 文档字典
            
        返回:
            评估指标字典
        """
        from bert_base.metrics import DPRMetrics
        
        # 计算检索指标
        metrics = DPRMetrics.calculate_retrieval_metrics(
            model=model,
            tokenizer=tokenizer,
            questions=eval_questions,
            ground_truth_doc_ids=eval_doc_ids,
            documents=documents,
            device=self.device,
            top_k=5
        )
        
        return metrics
        
    def cross_validate(self, train_questions, train_contexts, train_doc_ids, documents):
        """
        使用交叉验证训练模型
        
        参数:
            train_questions: 训练问题列表
            train_contexts: 训练上下文（正负例）列表
            train_doc_ids: 训练问题对应的正确文档ID列表
            documents: 文档字典
            
        返回:
            最佳模型和指标
        """
        # 从config中获取交叉验证的参数
        cv_config = self.dpr_config.get("cross_validation", {})
        n_folds = cv_config.get("n_folds", 5)
        
        # 如果n_folds <= 1，不进行交叉验证
        if n_folds <= 1:
            info(f"[DPRTrainer] 不使用交叉验证 (n_folds={n_folds})")
            return None, None
            
        info(f"[DPRTrainer] 开始{n_folds}折交叉验证")
        
        # 打乱数据
        indices = list(range(len(train_questions)))
        random.shuffle(indices)
        
        # 分割数据为n_folds份
        fold_size = len(indices) // n_folds
        folds = []
        for i in range(n_folds):
            start = i * fold_size
            end = (i+1) * fold_size if i < n_folds - 1 else len(indices)
            fold_indices = indices[start:end]
            folds.append(fold_indices)
        
        best_model = None
        best_tokenizer = None
        best_metrics = None
        best_val_score = -float('inf')
        
        # 进行交叉验证
        for fold in range(n_folds):
            info(f"[DPRTrainer] 开始第{fold+1}/{n_folds}折训练")
            
            # 划分训练集和验证集
            val_indices = folds[fold]
            train_indices = [idx for f in range(n_folds) if f != fold for idx in folds[f]]
            
            # 准备当前折的数据
            fold_train_questions = [train_questions[i] for i in train_indices]
            fold_train_contexts = [train_contexts[i] for i in train_indices]
            fold_val_questions = [train_questions[i] for i in val_indices]
            fold_val_doc_ids = [train_doc_ids[i] for i in val_indices]
            
            # 创建数据集和加载器
            tokenizer = BertTokenizer.from_pretrained(self.model_path)
            train_dataset = DPRDataset(fold_train_questions, fold_train_contexts, tokenizer, self.train_params["max_length"])
            
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=self.train_params["batch_size"],
                shuffle=True,
                num_workers=0
            )
            
            # 初始化模型
            question_encoder = BertModel.from_pretrained(self.model_path).to(self.device)
            context_encoder = BertModel.from_pretrained(self.model_path).to(self.device)
            
            # 训练当前折的模型
            optimizer = AdamW([
                {"params": question_encoder.parameters(), "lr": self.train_params["learning_rate"]},
                {"params": context_encoder.parameters(), "lr": self.train_params["learning_rate"]}
            ], lr=self.train_params["learning_rate"], weight_decay=self.train_params["weight_decay"])
            
            # 计算总的训练步数和学习率调度器
            total_steps = len(train_dataloader) * self.train_params["epochs"]
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.train_params["warmup_steps"],
                num_training_steps=total_steps
            )
            
            # 训练模型
            fold_best_loss = float('inf')
            fold_best_metrics = None
            
            for epoch in range(self.train_params["epochs"]):
                # 训练
                question_encoder.train()
                context_encoder.train()
                
                train_loss = 0.0
                train_steps = 0
                
                for batch in tqdm(train_dataloader, desc=f"训练 Fold {fold+1} Epoch {epoch+1}"):
                    # 处理批次数据
                    question_inputs = {
                        "input_ids": batch["question_input_ids"].to(self.device),
                        "attention_mask": batch["question_attention_mask"].to(self.device),
                    }
                    if batch["question_token_type_ids"] is not None:
                        question_inputs["token_type_ids"] = batch["question_token_type_ids"].to(self.device)
                    
                    positive_inputs = {
                        "input_ids": batch["positive_input_ids"].to(self.device),
                        "attention_mask": batch["positive_attention_mask"].to(self.device),
                    }
                    if batch["positive_token_type_ids"] is not None:
                        positive_inputs["token_type_ids"] = batch["positive_token_type_ids"].to(self.device)
                    
                    # 前向传播
                    optimizer.zero_grad()
                    
                    # 编码问题和正例文档
                    question_outputs = question_encoder(**question_inputs)
                    question_embeddings = question_outputs.last_hidden_state[:, 0, :]
                    
                    positive_outputs = context_encoder(**positive_inputs)
                    positive_embeddings = positive_outputs.last_hidden_state[:, 0, :]
                    
                    # 计算正例得分
                    positive_scores = torch.sum(question_embeddings * positive_embeddings, dim=1)
                    
                    # 处理负例
                    batch_size = len(batch["question_input_ids"])
                    
                    # 初始化损失
                    loss = 0.0
                    
                    # 对每个样本处理负例
                    for i in range(batch_size):
                        sample_negs = batch["negative_contexts"][i]
                        sample_num_negs = batch["num_negatives"][i]
                        
                        if sample_num_negs == 0:
                            continue
                        
                        # 收集负例输入
                        neg_input_ids = []
                        neg_attention_mask = []
                        neg_token_type_ids = []
                        
                        for j in range(sample_num_negs):
                            neg = sample_negs[j]
                            neg_input_ids.append(neg["input_ids"])
                            neg_attention_mask.append(neg["attention_mask"])
                            if neg["token_type_ids"] is not None:
                                neg_token_type_ids.append(neg["token_type_ids"])
                        
                        neg_inputs = {
                            "input_ids": torch.stack(neg_input_ids).to(self.device),
                            "attention_mask": torch.stack(neg_attention_mask).to(self.device),
                        }
                        if neg_token_type_ids:
                            neg_inputs["token_type_ids"] = torch.stack(neg_token_type_ids).to(self.device)
                        
                        # 编码负例
                        neg_outputs = context_encoder(**neg_inputs)
                        neg_embeddings = neg_outputs.last_hidden_state[:, 0, :]
                        
                        # 计算负例得分
                        neg_scores = torch.sum(question_embeddings[i].unsqueeze(0).expand(sample_num_negs, -1) * neg_embeddings, dim=1)
                        
                        # 计算损失
                        all_scores = torch.cat([positive_scores[i].unsqueeze(0), neg_scores])
                        target = torch.zeros(1, dtype=torch.long, device=self.device)
                        sample_loss = F.cross_entropy(all_scores.unsqueeze(0), target)
                        
                        loss += sample_loss
                    
                    # 平均损失
                    loss = loss / batch_size
                    
                    # 反向传播
                    loss.backward()
                    
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(question_encoder.parameters(), self.train_params["max_grad_norm"])
                    torch.nn.utils.clip_grad_norm_(context_encoder.parameters(), self.train_params["max_grad_norm"])
                    
                    # 更新参数
                    optimizer.step()
                    scheduler.step()
                    
                    # 更新损失
                    train_loss += loss.item()
                    train_steps += 1
                
                avg_train_loss = train_loss / train_steps
                info(f"[DPRTrainer] Fold {fold+1} Epoch {epoch+1} 训练损失: {avg_train_loss:.4f}")
                
                # 评估模型
                question_encoder.eval()
                metrics = self.evaluate_model(
                    model=question_encoder,
                    tokenizer=tokenizer,
                    eval_questions=fold_val_questions,
                    eval_doc_ids=fold_val_doc_ids,
                    documents=documents
                )
                
                # 记录指标
                info(f"[DPRTrainer] Fold {fold+1} Epoch {epoch+1} 验证指标: " + 
                     ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()]))
                
                # 使用MRR作为主要评估指标
                val_score = metrics["mrr"]
                
                # 保存最佳模型
                if val_score > fold_best_loss:
                    fold_best_loss = val_score
                    fold_best_metrics = metrics
                    info(f"[DPRTrainer] 找到当前折的最佳模型，MRR: {val_score:.4f}")
            
            # 比较当前折与最佳模型
            if fold_best_loss > best_val_score:
                best_val_score = fold_best_loss
                best_model = (question_encoder, context_encoder)
                best_tokenizer = tokenizer
                best_metrics = fold_best_metrics
                info(f"[DPRTrainer] 找到新的全局最佳模型，MRR: {best_val_score:.4f}")
        
        info(f"[DPRTrainer] 交叉验证完成，最佳MRR: {best_val_score:.4f}")
        info(f"[DPRTrainer] 最佳模型指标: " + ", ".join([f"{k}={v:.4f}" for k, v in best_metrics.items()]))
        
        return best_model, best_tokenizer
    
    def train(self):
        """训练DPR模型"""
        # 加载数据
        train_data, val_data = self.load_raw_data()
        documents = self.load_processed_documents()
        
        # 准备训练数据
        train_questions, train_contexts, val_questions, val_contexts = self.prepare_training_data(
            train_data, val_data, documents)
        
        # 提取训练集中的文档ID作为训练集的ground truth
        train_doc_ids = [item.get("document_id", "") for item in train_data 
                         if item.get("query", "") in train_questions]
        
        # 提取验证集中的文档ID作为验证集的ground truth
        val_doc_ids = [item.get("document_id", "") for item in val_data 
                       if item.get("query", "") in val_questions]
        
        info(f"[DPRTrainer] 准备了{len(train_questions)}条训练样本和{len(val_questions)}条验证样本")
        
        # 检查是否使用交叉验证
        use_cross_validation = self.dpr_config.get("cross_validation", {}).get("enabled", False)
        
        if use_cross_validation:
            info(f"[DPRTrainer] 使用交叉验证进行训练")
            best_model, best_tokenizer = self.cross_validate(
                train_questions, train_contexts, train_doc_ids, documents)
            
            if best_model:
                question_encoder, context_encoder = best_model
                tokenizer = best_tokenizer
            else:
                # 如果交叉验证未能产生有效模型，使用标准训练
                info(f"[DPRTrainer] 交叉验证未产生有效模型，使用标准训练")
                question_encoder, context_encoder, tokenizer = self.standard_train(
                    train_questions, train_contexts, val_questions, val_contexts, 
                    train_doc_ids, val_doc_ids, documents)
        else:
            # 使用标准训练方法
            info(f"[DPRTrainer] 使用标准训练方法")
            question_encoder, context_encoder, tokenizer = self.standard_train(
                train_questions, train_contexts, val_questions, val_contexts, 
                train_doc_ids, val_doc_ids, documents)
        
        # 创建输出目录
        os.makedirs(self.data_paths["model_output_dir"], exist_ok=True)
        
        # 保存最终模型
        info(f"[DPRTrainer] 保存最终模型...")
        question_encoder_dir = os.path.join(self.data_paths["model_output_dir"], "question_encoder")
        context_encoder_dir = os.path.join(self.data_paths["model_output_dir"], "context_encoder")
        os.makedirs(question_encoder_dir, exist_ok=True)
        os.makedirs(context_encoder_dir, exist_ok=True)
        
        question_encoder.save_pretrained(question_encoder_dir)
        context_encoder.save_pretrained(context_encoder_dir)
        tokenizer.save_pretrained(self.data_paths["model_output_dir"])
        
        # 训练完成，构建向量索引
        info(f"[DPRTrainer] 训练完成，开始构建向量索引...")
        self.build_document_index(context_encoder, tokenizer, documents)
        
        info(f"[DPRTrainer] DPR模型训练和索引构建完成！")
        return self.data_paths["model_output_dir"]
    
    def standard_train(self, train_questions, train_contexts, val_questions, val_contexts, train_doc_ids, val_doc_ids, documents):
        """标准训练方法（无交叉验证）"""
        # 初始化模型和tokenizer
        info(f"[DPRTrainer] 初始化模型和tokenizer，使用预训练模型: {self.model_path}")
        tokenizer = BertTokenizer.from_pretrained(self.model_path)
        
        # 初始化问题编码器和文档编码器（使用相同的预训练模型）
        question_encoder = BertModel.from_pretrained(self.model_path)
        context_encoder = BertModel.from_pretrained(self.model_path)
        
        # 将模型移到设备上
        question_encoder = question_encoder.to(self.device)
        context_encoder = context_encoder.to(self.device)
        
        # 创建数据集和数据加载器
        train_dataset = DPRDataset(train_questions, train_contexts, tokenizer, self.train_params["max_length"])
        val_dataset = DPRDataset(val_questions, val_contexts, tokenizer, self.train_params["max_length"])
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.train_params["batch_size"],
            shuffle=True,
            num_workers=0  # 在Windows上设置为0以避免多进程问题
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.train_params["batch_size"],
            shuffle=False,
            num_workers=0  # 在Windows上设置为0以避免多进程问题
        )
        
        # 初始化优化器
        optimizer = AdamW([
            {"params": question_encoder.parameters(), "lr": self.train_params["learning_rate"]},
            {"params": context_encoder.parameters(), "lr": self.train_params["learning_rate"]}
        ], lr=self.train_params["learning_rate"], weight_decay=self.train_params["weight_decay"])
        
        # 计算总的训练步数
        total_steps = len(train_dataloader) * self.train_params["epochs"]
        
        # 初始化学习率调度器
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.train_params["warmup_steps"],
            num_training_steps=total_steps
        )
        
        # 训练模型
        info(f"[DPRTrainer] 开始标准训练，共{self.train_params['epochs']}个epoch，设备: {self.device}")
        
        best_val_loss = float('inf')
        best_val_metrics = None
        
        for epoch in range(self.train_params["epochs"]):
            info(f"[DPRTrainer] Epoch {epoch+1}/{self.train_params['epochs']}")
            
            # 训练阶段
            question_encoder.train()
            context_encoder.train()
            
            train_loss = 0.0
            train_steps = 0
            
            train_progress_bar = tqdm(train_dataloader, desc=f"训练 Epoch {epoch+1}")
            for batch in train_progress_bar:
                # 提取批次数据
                question_inputs = {
                    "input_ids": batch["question_input_ids"].to(self.device),
                    "attention_mask": batch["question_attention_mask"].to(self.device),
                }
                if batch["question_token_type_ids"] is not None:
                    question_inputs["token_type_ids"] = batch["question_token_type_ids"].to(self.device)
                
                positive_inputs = {
                    "input_ids": batch["positive_input_ids"].to(self.device),
                    "attention_mask": batch["positive_attention_mask"].to(self.device),
                }
                if batch["positive_token_type_ids"] is not None:
                    positive_inputs["token_type_ids"] = batch["positive_token_type_ids"].to(self.device)
                
                # 前向传播
                optimizer.zero_grad()
                
                # 编码问题
                question_outputs = question_encoder(**question_inputs)
                question_embeddings = question_outputs.last_hidden_state[:, 0, :]  # [CLS]向量
                
                # 编码正例文档
                positive_outputs = context_encoder(**positive_inputs)
                positive_embeddings = positive_outputs.last_hidden_state[:, 0, :]  # [CLS]向量
                
                # 计算正例得分（点积）
                positive_scores = torch.sum(question_embeddings * positive_embeddings, dim=1)
                
                # 处理负例
                batch_size = len(batch["question_input_ids"])
                max_negatives = max(batch["num_negatives"])
                
                # 初始化损失
                loss = 0.0
                
                # 对每个样本单独处理其负例
                for i in range(batch_size):
                    sample_negs = batch["negative_contexts"][i]
                    sample_num_negs = batch["num_negatives"][i]
                    
                    if sample_num_negs == 0:
                        continue
                    
                    # 收集该样本的所有负例输入
                    neg_input_ids = []
                    neg_attention_mask = []
                    neg_token_type_ids = []
                    
                    for j in range(sample_num_negs):
                        neg = sample_negs[j]
                        neg_input_ids.append(neg["input_ids"])
                        neg_attention_mask.append(neg["attention_mask"])
                        if neg["token_type_ids"] is not None:
                            neg_token_type_ids.append(neg["token_type_ids"])
                    
                    neg_inputs = {
                        "input_ids": torch.stack(neg_input_ids).to(self.device),
                        "attention_mask": torch.stack(neg_attention_mask).to(self.device),
                    }
                    if neg_token_type_ids:
                        neg_inputs["token_type_ids"] = torch.stack(neg_token_type_ids).to(self.device)
                    
                    # 编码负例
                    neg_outputs = context_encoder(**neg_inputs)
                    neg_embeddings = neg_outputs.last_hidden_state[:, 0, :]  # [CLS]向量
                    
                    # 计算负例得分（点积）
                    neg_scores = torch.sum(question_embeddings[i].unsqueeze(0).expand(sample_num_negs, -1) * neg_embeddings, dim=1)
                    
                    # 将正例得分与负例得分拼接
                    all_scores = torch.cat([positive_scores[i].unsqueeze(0), neg_scores])
                    
                    # 计算交叉熵损失（索引0是正例）
                    target = torch.zeros(all_scores.size(0), dtype=torch.long, device=self.device)
                    sample_loss = F.cross_entropy(all_scores.unsqueeze(0), target.unsqueeze(0))
                    
                    loss += sample_loss
                
                # 平均损失
                loss = loss / batch_size
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(question_encoder.parameters(), self.train_params["max_grad_norm"])
                torch.nn.utils.clip_grad_norm_(context_encoder.parameters(), self.train_params["max_grad_norm"])
                
                # 更新参数
                optimizer.step()
                scheduler.step()
                
                # 更新损失
                train_loss += loss.item()
                train_steps += 1
                
                # 更新进度条
                train_progress_bar.set_postfix({"loss": train_loss / train_steps})
            
            avg_train_loss = train_loss / train_steps
            info(f"[DPRTrainer] Epoch {epoch+1} 训练损失: {avg_train_loss:.4f}")
            
            # 验证阶段 - 首先使用传统方法计算验证损失
            question_encoder.eval()
            context_encoder.eval()
            
            val_loss = 0.0
            val_steps = 0
            
            val_progress_bar = tqdm(val_dataloader, desc=f"验证 Epoch {epoch+1}")
            with torch.no_grad():
                for batch in val_progress_bar:
                    # 提取批次数据
                    question_inputs = {
                        "input_ids": batch["question_input_ids"].to(self.device),
                        "attention_mask": batch["question_attention_mask"].to(self.device),
                    }
                    if batch["question_token_type_ids"] is not None:
                        question_inputs["token_type_ids"] = batch["question_token_type_ids"].to(self.device)
                    
                    positive_inputs = {
                        "input_ids": batch["positive_input_ids"].to(self.device),
                        "attention_mask": batch["positive_attention_mask"].to(self.device),
                    }
                    if batch["positive_token_type_ids"] is not None:
                        positive_inputs["token_type_ids"] = batch["positive_token_type_ids"].to(self.device)
                    
                    # 编码问题
                    question_outputs = question_encoder(**question_inputs)
                    question_embeddings = question_outputs.last_hidden_state[:, 0, :]  # [CLS]向量
                    
                    # 编码正例文档
                    positive_outputs = context_encoder(**positive_inputs)
                    positive_embeddings = positive_outputs.last_hidden_state[:, 0, :]  # [CLS]向量
                    
                    # 计算正例得分（点积）
                    positive_scores = torch.sum(question_embeddings * positive_embeddings, dim=1)
                    
                    # 处理负例
                    batch_size = len(batch["question_input_ids"])
                    
                    # 初始化损失
                    loss = 0.0
                    
                    # 对每个样本单独处理其负例
                    for i in range(batch_size):
                        sample_negs = batch["negative_contexts"][i]
                        sample_num_negs = batch["num_negatives"][i]
                        
                        if sample_num_negs == 0:
                            continue
                        
                        # 收集该样本的所有负例输入
                        neg_input_ids = []
                        neg_attention_mask = []
                        neg_token_type_ids = []
                        
                        for j in range(sample_num_negs):
                            neg = sample_negs[j]
                            neg_input_ids.append(neg["input_ids"])
                            neg_attention_mask.append(neg["attention_mask"])
                            if neg["token_type_ids"] is not None:
                                neg_token_type_ids.append(neg["token_type_ids"])
                        
                        neg_inputs = {
                            "input_ids": torch.stack(neg_input_ids).to(self.device),
                            "attention_mask": torch.stack(neg_attention_mask).to(self.device),
                        }
                        if neg_token_type_ids:
                            neg_inputs["token_type_ids"] = torch.stack(neg_token_type_ids).to(self.device)
                        
                        # 编码负例
                        neg_outputs = context_encoder(**neg_inputs)
                        neg_embeddings = neg_outputs.last_hidden_state[:, 0, :]  # [CLS]向量
                        
                        # 计算负例得分（点积）
                        neg_scores = torch.sum(question_embeddings[i].unsqueeze(0).expand(sample_num_negs, -1) * neg_embeddings, dim=1)
                        
                        # 将正例得分与负例得分拼接
                        all_scores = torch.cat([positive_scores[i].unsqueeze(0), neg_scores])
                        
                        # 计算交叉熵损失（索引0是正例）
                        target = torch.zeros(all_scores.size(0), dtype=torch.long, device=self.device)
                        sample_loss = F.cross_entropy(all_scores.unsqueeze(0), target.unsqueeze(0))
                        
                        loss += sample_loss
                    
                    # 平均损失
                    loss = loss / batch_size
                    
                    # 更新损失
                    val_loss += loss.item()
                    val_steps += 1
                    
                    # 更新进度条
                    val_progress_bar.set_postfix({"loss": val_loss / val_steps})
            
            avg_val_loss = val_loss / val_steps
            
            # 计算验证集上的指标
            info(f"[DPRTrainer] 计算验证集上的评估指标...")
            val_metrics = self.evaluate_model(
                model=question_encoder,
                tokenizer=tokenizer,
                eval_questions=val_questions,
                eval_doc_ids=val_doc_ids,
                documents=documents
            )
            
            # 计算训练集上的指标
            info(f"[DPRTrainer] 计算训练集上的评估指标...")
            train_metrics = self.evaluate_model(
                model=question_encoder,
                tokenizer=tokenizer,
                eval_questions=train_questions[:min(len(train_questions), 1000)],  # 取部分训练样本计算指标
                eval_doc_ids=train_doc_ids[:min(len(train_doc_ids), 1000)],
                documents=documents
            )
            
            # 输出评估指标
            info(f"[DPRTrainer] Epoch {epoch+1} 训练指标: " + 
                 ", ".join([f"{k}={v:.4f}" for k, v in train_metrics.items()]))
            info(f"[DPRTrainer] Epoch {epoch+1} 验证损失: {avg_val_loss:.4f}, 验证指标: " + 
                 ", ".join([f"{k}={v:.4f}" for k, v in val_metrics.items()]))
            
            # 保存最佳模型（使用MRR作为主要评估指标）
            val_score = val_metrics.get("mrr", 0.0)
            if val_score > best_val_loss:
                best_val_loss = val_score
                best_val_metrics = val_metrics
                info(f"[DPRTrainer] 找到新的最佳模型，验证MRR: {best_val_loss:.4f}，保存模型...")
        
        info(f"[DPRTrainer] 训练完成，最佳验证MRR: {best_val_loss:.4f}")
        info(f"[DPRTrainer] 最佳模型指标: " + ", ".join([f"{k}={v:.4f}" for k, v in best_val_metrics.items()]))
        
        return question_encoder, context_encoder, tokenizer
    
    def build_document_index(self, context_encoder, tokenizer, documents):
        """构建文档向量索引"""
        info(f"[DPRTrainer] 开始为所有文档构建向量索引...")
        
        # 初始化DPR检索器
        dpr = DensePassageRetriever(self.config)
        dpr.context_encoder = context_encoder
        dpr.tokenizer = tokenizer
        dpr.device = self.device
        
        # 获取所有文档文本和对应的ID
        doc_texts = []
        doc_ids = []
        
        for doc_id, text in tqdm(documents.items(), desc="准备文档"):
            doc_texts.append(text)
            doc_ids.append(doc_id)
        
        # 构建文档向量
        batch_size = self.dpr_config.get("index_batch_size", 8)
        info(f"[DPRTrainer] 开始为{len(doc_texts)}个文档生成向量表示，批大小: {batch_size}...")
        
        doc_embeddings = []
        for i in tqdm(range(0, len(doc_texts), batch_size), desc="编码文档"):
            batch_texts = doc_texts[i:i+batch_size]
            
            # 使用DPR编码器获取文档向量
            batch_embeddings = dpr.encode_contexts(batch_texts, batch_size=batch_size)
            doc_embeddings.append(batch_embeddings)
        
        # 合并所有文档向量
        all_embeddings = np.vstack(doc_embeddings)
        info(f"[DPRTrainer] 已生成{all_embeddings.shape[0]}个文档向量，每个维度: {all_embeddings.shape[1]}")
        
        # 构建FAISS索引
        index_type = self.dpr_config.get("index_type", "Flat")
        info(f"[DPRTrainer] 使用{index_type}索引类型构建FAISS索引...")
        
        faiss_index = dpr.build_faiss_index(all_embeddings, index_type=index_type)
        
        # 保存索引
        dpr.faiss_index = faiss_index
        dpr.doc_ids = doc_ids
        
        index_dir = os.path.join(self.data_paths["model_output_dir"], "index")
        info(f"[DPRTrainer] 保存索引到{index_dir}...")
        
        dpr.save_index(index_dir, overwrite=True)
        
        info(f"[DPRTrainer] 索引构建和保存完成！")


def run_dpr_training(config_path="config.yaml"):
    """运行DPR训练主函数"""
    info(f"[DPR] 启动DPR训练...")
    
    # 创建训练器并开始训练
    trainer = DPRTrainer(config_path)
    
    # 训练模型和构建索引
    model_dir = trainer.train()
    
    info(f"[DPR] DPR训练完成，模型已保存到: {model_dir}")
    return True

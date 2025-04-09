import os
import torch
import logging
import random
import numpy as np
# 导入Windows平台补丁
from bert_base.patch_windows import *
from tqdm import tqdm
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup

from bert_base.model import DPRModel
from bert_base.dataset import create_dataloader

import warnings
warnings.filterwarnings("ignore")

def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def prepare_dpr_data(config):
    """
    准备DPR训练所需的数据
    
    Args:
        config: 配置字典
    
    Returns:
        bool: 数据准备是否成功
    """
    # 导入数据处理器
    try:
        from bert_base.dpr_processor import DocumentPipeline, DPRDataProcessor
    except ImportError:
        logging.error("无法导入数据处理模块，请确保bert_base.dpr_processor模块已安装")
        return False
    
    # 获取数据路径
    data_config = config.get("data", {})
    data_output_dir = data_config.get("data_output_dir", "./data/processed_data/")
    if not data_output_dir.endswith("/"):
        data_output_dir += "/"
    
    # 确保路径存在
    os.makedirs(data_output_dir, exist_ok=True)
    
    # 检查处理后的文本文件是否存在
    processed_file = os.path.join(data_output_dir, "processed_plain.jsonl")
    
    # 如果处理后的文件不存在，运行数据处理流程
    if not os.path.exists(processed_file):
        logging.info("缺少处理后的文件，请先运行数据处理流程")
        logging.info("可以使用以下命令：python main.py --mode process")
        return False
    
    # 获取DPR训练参数
    dpr_config = config.get("bert_base", {}).get("train", {})
    chunk_size = dpr_config.get("max_ctx_length", 2048)
    chunk_overlap = dpr_config.get("chunk_overlap", 100)
    n_negatives = dpr_config.get("n_negatives", 3)
    max_workers = dpr_config.get("num_workers", 4)
    
    # 准备DPR训练数据
    logging.info("开始准备DPR训练数据...")
    
    try:
        # 创建DPR数据处理器
        processor = DPRDataProcessor(data_dir=os.path.dirname(data_output_dir))
        
        # 执行完整的数据处理流程
        logging.info("加载文档...")
        processor.load_documents()
        
        logging.info("加载问答数据...")
        processor.load_qa_data()
        
        logging.info("分块文档...")
        processor.chunk_documents(chunk_size=chunk_size, overlap=chunk_overlap)
        
        logging.info("构建TF-IDF索引...")
        processor.build_tfidf_index()
        
        logging.info("准备DPR训练数据...")
        train_examples, val_examples = processor.prepare_dpr_training_data(
            n_negatives=n_negatives,
            max_workers=max_workers
        )
        
        logging.info("保存DPR训练数据...")
        processor.save_dpr_data(train_examples, val_examples)
        
        logging.info("DPR训练数据准备完成")
        return True
    
    except Exception as e:
        logging.error(f"准备DPR训练数据时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def train(config):
    """训练DPR模型"""
    # 获取训练配置
    train_config = config.get("bert_base", {}).get("train", {})
    
    # 设置随机种子
    seed = config.get("system", {}).get("seed", 42)
    set_seed(seed)
    
    # 获取路径配置
    paths_config = config.get("paths", {})
    model_root = paths_config.get("model_root", "./models/")
    cache_dir = paths_config.get("cache_dir", "./cache/")
    
    # 创建输出目录
    output_dir = os.path.join(model_root, "bert_base")
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取数据路径
    data_config = config.get("data", {})
    data_output_dir = data_config.get("data_output_dir", "./data/processed_data/")
    
    # 数据文件路径
    train_file = os.path.join(data_output_dir, "dpr_train.json")
    val_file = os.path.join(data_output_dir, "dpr_val.json")
    
    # 检查训练和验证文件是否存在
    if not (os.path.exists(train_file) and os.path.exists(val_file)):
        logging.error(f"训练数据文件不存在: {train_file} 或 {val_file}")
        logging.info("请先使用 --mode process 进行数据处理，或者确保已生成DPR训练数据")
        return 0, 0
    
    # 获取模型配置
    model_name_or_path = train_config.get("model_name_or_path", "nomic-ai/nomic-bert-2048")
    shared_weights = train_config.get("shared_weights", False)
    temperature = train_config.get("temperature", 0.05)
    
    # 训练参数
    num_train_epochs = train_config.get("num_train_epochs", 5)
    batch_size = train_config.get("batch_size", 2)
    learning_rate = train_config.get("learning_rate", 3e-5)
    weight_decay = train_config.get("weight_decay", 0.01)
    max_query_length = train_config.get("max_query_length", 128)
    max_ctx_length = train_config.get("max_ctx_length", 2048)
    warmup_steps = train_config.get("warmup_steps", 0)
    adam_epsilon = train_config.get("adam_epsilon", 1e-8)
    eval_steps = train_config.get("eval_steps", 0)
    gradient_accumulation_steps = train_config.get("gradient_accumulation_steps", 2)
    num_workers = train_config.get("num_workers", 4)
    
    # 设备
    device_name = config.get("system", {}).get("device", "cuda")
    device = torch.device(device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    logging.info(f"使用设备: {device}")
    
    # 加载tokenizer
    logging.info(f"加载tokenizer: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, cache_dir=cache_dir)
    
    # 加载数据
    logging.info("创建训练数据加载器...")
    train_dataloader = create_dataloader(
        data_file=train_file,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_query_length=max_query_length,
        max_ctx_length=max_ctx_length,
        is_training=True,
        shuffle=True,
        num_workers=num_workers
    )
    
    # 加载验证数据
    logging.info("创建验证数据加载器...")
    val_dataloader = create_dataloader(
        data_file=val_file,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_query_length=max_query_length,
        max_ctx_length=max_ctx_length,
        is_training=True,  # 仍然需要负样本来计算对比损失
        shuffle=False,
        num_workers=num_workers
    )
    
    # 创建模型
    logging.info(f"创建DPR模型，使用预训练模型: {model_name_or_path}")
    model = DPRModel(
        query_encoder_name=model_name_or_path,
        ctx_encoder_name=model_name_or_path,
        shared_weights=shared_weights,
        temperature=temperature
    )
    model.to(device)
    
    # 准备优化器
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    # 计算总训练步数
    if gradient_accumulation_steps > 1:
        # 考虑梯度累积
        t_total = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs
    else:
        t_total = len(train_dataloader) * num_train_epochs
    
    # 创建优化器和学习率调度器
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=t_total
    )
    
    # 是否使用混合精度训练
    scaler = torch.amp.GradScaler() if torch.cuda.is_available() else None
    
    # 训练循环
    logging.info("***** 开始训练 *****")
    logging.info(f"  每个批次样本数 = {batch_size}")
    logging.info(f"  梯度累积步数 = {gradient_accumulation_steps}")
    logging.info(f"  总训练步数 = {t_total}")
    logging.info(f"  最大上下文长度 = {max_ctx_length}")
    if eval_steps > 0:
        logging.info(f"  每 {eval_steps} 步进行一次评估")
    else:
        logging.info(f"  仅在每个epoch结束后进行评估")
    
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(num_train_epochs):
        logging.info(f"开始 Epoch {epoch+1}/{num_train_epochs}")
        model.train()
        epoch_loss = 0.0
        
        for step, batch in enumerate(tqdm(train_dataloader, desc="训练")):
            # 将数据移到设备上
            batch = {k: v.to(device) if v is not None else None for k, v in batch.items()}
            
            # 添加调试日志，输出一些批次信息以便追踪
            if step == 0:
                logging.info(f"批次形状: query_input_ids={batch['query_input_ids'].shape}, "
                           f"pos_ctx_input_ids={batch['pos_ctx_input_ids'].shape}, "
                           f"neg_ctx_input_ids={batch['neg_ctx_input_ids'].shape}")
            
            # 使用混合精度训练
            if scaler is not None:
                with torch.amp.autocast(device_type='cuda'):
                    # 前向传播
                    query_vectors, pos_ctx_vectors, neg_ctx_vectors = model(
                        query_input_ids=batch["query_input_ids"],
                        query_attention_mask=batch["query_attention_mask"],
                        query_token_type_ids=batch["query_token_type_ids"],
                        pos_ctx_input_ids=batch["pos_ctx_input_ids"],
                        pos_ctx_attention_mask=batch["pos_ctx_attention_mask"],
                        pos_ctx_token_type_ids=batch["pos_ctx_token_type_ids"],
                        neg_ctx_input_ids=batch["neg_ctx_input_ids"],
                        neg_ctx_attention_mask=batch["neg_ctx_attention_mask"],
                        neg_ctx_token_type_ids=batch["neg_ctx_token_type_ids"]
                    )
                    
                    # 计算损失
                    loss = model.compute_loss(query_vectors, pos_ctx_vectors, neg_ctx_vectors)
                    
                    # 梯度累积
                    loss = loss / gradient_accumulation_steps
                
                # 反向传播
                scaler.scale(loss).backward()
                
                # 每隔指定步数更新参数
                if (step + 1) % gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
            else:
                # 前向传播
                query_vectors, pos_ctx_vectors, neg_ctx_vectors = model(
                    query_input_ids=batch["query_input_ids"],
                    query_attention_mask=batch["query_attention_mask"],
                    query_token_type_ids=batch["query_token_type_ids"],
                    pos_ctx_input_ids=batch["pos_ctx_input_ids"],
                    pos_ctx_attention_mask=batch["pos_ctx_attention_mask"],
                    pos_ctx_token_type_ids=batch["pos_ctx_token_type_ids"],
                    neg_ctx_input_ids=batch["neg_ctx_input_ids"],
                    neg_ctx_attention_mask=batch["neg_ctx_attention_mask"],
                    neg_ctx_token_type_ids=batch["neg_ctx_token_type_ids"]
                )
                
                # 计算损失
                loss = model.compute_loss(query_vectors, pos_ctx_vectors, neg_ctx_vectors)
                
                # 梯度累积
                loss = loss / gradient_accumulation_steps
            
                # 反向传播
                loss.backward()
                
                # 每隔指定步数更新参数
                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
            
            epoch_loss += loss.item() * gradient_accumulation_steps
            
            # 每隔eval_steps步进行一次评估
            if eval_steps > 0 and global_step > 0 and global_step % eval_steps == 0:
                val_loss = evaluate(config, model, val_dataloader, device)
                
                # 如果验证损失更好，保存模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    logging.info(f"保存最佳模型，验证损失: {val_loss:.4f}")
                    save_model(config, model, tokenizer, f"best_model_step_{global_step}")
                
                model.train()  # 回到训练模式
        
        # 处理最后一个批次的梯度
        if (step + 1) % gradient_accumulation_steps != 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
        
        # 计算平均损失
        avg_loss = epoch_loss / len(train_dataloader)
        logging.info(f"Epoch {epoch+1} 平均损失: {avg_loss:.4f}")
        
        # 每个epoch结束后评估
        if eval_steps <= 0:  # 只有在不进行中间评估时，才在epoch结束后评估
            logging.info("进行验证...")
            val_loss = evaluate(config, model, val_dataloader, device)
            
            # 如果验证损失更好，保存模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                logging.info(f"保存最佳模型，验证损失: {val_loss:.4f}")
                save_model(config, model, tokenizer, "best_model")
        
        # 保存每个epoch的模型
        logging.info(f"保存Epoch {epoch+1}模型")
        save_model(config, model, tokenizer, f"epoch_{epoch+1}")
    
    # 保存最终模型
    logging.info("保存最终模型")
    save_model(config, model, tokenizer, "final_model")
    
    return global_step, best_val_loss

def evaluate(config, model, dataloader, device):
    """评估模型"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="评估"):
            # 将数据移到设备上
            batch = {k: v.to(device) if v is not None else None for k, v in batch.items()}
            
            # 前向传播
            query_vectors, pos_ctx_vectors, neg_ctx_vectors = model(
                query_input_ids=batch["query_input_ids"],
                query_attention_mask=batch["query_attention_mask"],
                query_token_type_ids=batch["query_token_type_ids"],
                pos_ctx_input_ids=batch["pos_ctx_input_ids"],
                pos_ctx_attention_mask=batch["pos_ctx_attention_mask"],
                pos_ctx_token_type_ids=batch["pos_ctx_token_type_ids"],
                neg_ctx_input_ids=batch["neg_ctx_input_ids"],
                neg_ctx_attention_mask=batch["neg_ctx_attention_mask"],
                neg_ctx_token_type_ids=batch["neg_ctx_token_type_ids"]
            )
            
            # 计算损失
            loss = model.compute_loss(query_vectors, pos_ctx_vectors, neg_ctx_vectors)
            total_loss += loss.item()
    
    # 计算平均损失
    avg_loss = total_loss / len(dataloader)
    logging.info(f"验证损失: {avg_loss:.4f}")
    
    return avg_loss

def save_model(config, model, tokenizer, prefix):
    """保存模型和分词器"""
    paths_config = config.get("paths", {})
    model_root = paths_config.get("model_root", "./models/")
    
    output_dir = os.path.join(model_root, "bert_base", prefix)
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存编码器
    model.save_encoders(output_dir)
    
    # 保存分词器
    tokenizer.save_pretrained(output_dir)
    
    # 保存训练配置
    train_config = config.get("bert_base", {}).get("train", {})
    with open(os.path.join(output_dir, "training_config.txt"), "w") as f:
        for key, value in train_config.items():
            f.write(f"{key}: {value}\n")

def train_bert_base_model(config):
    """启动BERT-Base模型训练的主函数"""
    train_mode = config.get("debug", {}).get("train_mode", False)
    
    if not train_mode:
        logging.info("训练模式未启用，跳过BERT-Base模型训练")
        return
    
    # 准备数据
    data_config = config.get("data", {})
    data_output_dir = data_config.get("data_output_dir", "./data/processed_data/")
    train_file = os.path.join(data_output_dir, "dpr_train.json")
    val_file = os.path.join(data_output_dir, "dpr_val.json")
    
    # 检查训练文件是否存在
    if not (os.path.exists(train_file) and os.path.exists(val_file)):
        logging.info("DPR训练数据不存在，尝试自动准备...")
        success = prepare_dpr_data(config)
        if not success:
            logging.error("无法自动准备DPR训练数据，请先运行数据处理步骤")
            logging.info("可以使用以下命令：python main.py --mode process")
            logging.info("然后确保processed_plain.jsonl文件已生成")
            logging.info("或者手动创建dpr_train.json和dpr_val.json文件")
            return
    
    logging.info("开始BERT-Base模型训练...")
    global_step, best_val_loss = train(config)
    logging.info(f"训练完成，总步数: {global_step}，最佳验证损失: {best_val_loss:.4f}")
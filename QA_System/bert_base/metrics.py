import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch

class DPRMetrics:
    """DPR模型评估指标计算工具类"""
    
    @staticmethod
    def calculate_retrieval_metrics(model, tokenizer, questions, ground_truth_doc_ids, documents, device, top_k=5):
        """
        计算检索模型的评估指标
        
        参数:
            model: 问题编码器模型
            tokenizer: 分词器
            questions: 问题列表
            ground_truth_doc_ids: 标准答案文档ID列表
            documents: 文档字典 {doc_id: text}
            device: 运行设备
            top_k: 检索的文档数量
            
        返回:
            包含各项指标的字典
        """
        model.eval()
        all_doc_ids = list(documents.keys())
        all_doc_texts = [documents[doc_id] for doc_id in all_doc_ids]
        
        # 编码所有文档
        doc_embeddings = []
        batch_size = 16
        for i in range(0, len(all_doc_texts), batch_size):
            batch = all_doc_texts[i:min(i+batch_size, len(all_doc_texts))]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                doc_embeddings.append(embeddings)
        
        doc_embeddings = np.vstack(doc_embeddings)
        
        # 计算评估指标
        hits_at_k = 0
        reciprocal_ranks = []
        precisions = []
        recalls = []
        
        for question, gt_doc_id in zip(questions, ground_truth_doc_ids):
            # 编码问题
            inputs = tokenizer(
                question,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                query_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            # 计算相似度得分
            scores = np.dot(query_embedding, doc_embeddings.T)[0]
            top_indices = np.argsort(-scores)[:top_k]
            retrieved_doc_ids = [all_doc_ids[idx] for idx in top_indices]
            
            # 计算hits@k
            if gt_doc_id in retrieved_doc_ids:
                hits_at_k += 1
            
            # 计算MRR
            if gt_doc_id in retrieved_doc_ids:
                rank = retrieved_doc_ids.index(gt_doc_id) + 1
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)
            
            # 计算精确率和召回率
            relevant = [1 if doc_id == gt_doc_id else 0 for doc_id in retrieved_doc_ids]
            precisions.append(sum(relevant) / len(relevant))
            recalls.append(1.0 if gt_doc_id in retrieved_doc_ids else 0.0)
        
        # 计算平均指标
        hit_rate = hits_at_k / len(questions)
        mrr = np.mean(reciprocal_ranks)
        precision = np.mean(precisions)
        recall = np.mean(recalls)
        
        return {
            "hit_rate@k": hit_rate,
            "mrr": mrr,
            "precision": precision,
            "recall": recall
        }
    
    @staticmethod
    def calculate_contrastive_metrics(query_embeddings, pos_embeddings, neg_embeddings):
        """
        计算对比学习的评估指标
        
        参数:
            query_embeddings: 问题嵌入向量
            pos_embeddings: 正例文档嵌入向量
            neg_embeddings: 负例文档嵌入向量
            
        返回:
            包含各项指标的字典
        """
        # 计算正例得分
        pos_scores = torch.sum(query_embeddings * pos_embeddings, dim=1)
        
        # 计算负例得分
        neg_scores_list = []
        for i in range(neg_embeddings.size(0)):
            neg_score = torch.sum(query_embeddings[i].unsqueeze(0) * neg_embeddings[i], dim=1)
            neg_scores_list.append(neg_score)
        
        # 计算平均正负例得分差异
        avg_pos_score = torch.mean(pos_scores).item()
        avg_neg_score = torch.mean(torch.cat(neg_scores_list)).item()
        score_diff = avg_pos_score - avg_neg_score
        
        # 计算分类准确率 (正例得分应高于负例)
        correct = 0
        total = 0
        
        for i in range(len(query_embeddings)):
            pos_score = pos_scores[i]
            neg_score = neg_scores_list[i]
            if pos_score > torch.max(neg_score):
                correct += 1
            total += 1
        
        accuracy = correct / total if total > 0 else 0
        
        return {
            "avg_pos_score": avg_pos_score,
            "avg_neg_score": avg_neg_score,
            "score_diff": score_diff,
            "accuracy": accuracy
        } 
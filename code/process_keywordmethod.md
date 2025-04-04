# Document Retrieval System (BM25 + TF-IDF Hybrid)

## 结构

- `document_process.py`: 文档预处理脚本  
  - 功能：读取原始 `document.jsonl`，提取纯文本并构建 BM25 与 TF-IDF 模型。  
  - 输出：
    - `processed_plain.jsonl`: 预处理好的纯文本文档  
    - `plain_bm25.pkl`: 保存的 BM25 模型  
    - `plain_tfidf.pkl`: 保存的 TF-IDF 模型  

- `query_with_Keywords.py`: 查询处理与文档检索脚本  
  - 模式 1（批量模式）：
    - 输入：标准格式的问题文件（如老师提供的验证集 `val.jsonl`）  
    - 输出：包含预测结果的 `.jsonl` 文件（前 5 个最相关文档 ID + 简要答案）  

  - 模式 2（单个查询）：
    - 手动输入单个问题  
    - 输出：最相关的前 5 个文档 ID 及简单预测答案，支持 JSON 格式展示  

---

## 实验结果（BM25 与 TF-IDF 加权混合）

我们对不同 BM25 与 TF-IDF 加权组合进行检索性能测试，评估指标为：

- **Recall@5**：前 5 个检索结果中是否包含正确答案文档  
- **MRR@5**（Mean Reciprocal Rank）：正确文档出现的排名倒数平均值  

### 使用预处理文档的结果：

| 权重组合              | Recall@5 | MRR@5   |
|----------------------|----------|---------|
| **1.0 BM25 / 0.0 TF** | 0.7210   | 0.5728  |
| **0.8 BM25 / 0.2 TF** | 0.7930   | 0.6339  |
| **0.7 BM25 / 0.3 TF** | **0.8020** | **0.6283** |
| 0.5 BM25 / 0.5 TF     | 0.7850   | 0.6006  |
| 0.3 BM25 / 0.7 TF     | 0.7370   | 0.5544  |
| 0.0 BM25 / 1.0 TF     | 0.6930   | 0.5128  |

### 未经预处理（Raw Document）的结果：

| 权重组合              | Recall@5 | MRR@5   |
|----------------------|----------|---------|
| **0.7 BM25 / 0.3 TF** | 0.7130   | 0.5291  |
| 1.0 BM25 / 0.0 TF     | 0.6080   | 0.4529  |
| 0.0 BM25 / 1.0 TF     | 0.6120   | 0.4355  |

---

## 当前结论

- **无论是否预处理**，单独使用 BM25 或 TF-IDF 的性能都不如二者加权融合。  
- **使用预处理文档时的最优组合**为 **0.7 BM25 + 0.3 TF-IDF**，达到：
  - Recall@5: **0.8020**
  - MRR@5: **0.6283**
- **未经预处理时性能整体下降**
- 在实际部署中推荐：
  - 对原始 HTML/JSON 文档进行清洗，提取纯文本再建模  
  - 使用 Hybrid 模式，默认设置 `alpha = 0.7`

---

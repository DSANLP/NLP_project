import numpy as np
import faiss
import os

class FaissSaver:
    '''
    input:
    id: Every unique id of documents List[str]
    vector: Embedding vector of documents List[List[float]] [MUST BE NORMALIZED]
    request: Retrieval Method Request of documents str
    save_path: Path to save the index  str

    output:
    index_file: index file
    '''
    def __init__(self, id, vector, request, save_path):
        self.id = id
        self.vector = vector
        self.request = request
        self.save_path = save_path

    def save(self):
        # 创建保存目录
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        
        int_ids = np.array([int(s) for s in self.id], dtype=np.int64)
        vecs = np.array(self.vector, dtype=np.float32)
        index_raw = faiss.IndexIDMap(faiss.IndexFlatIP(vecs.shape[1]))
        index_raw.add_with_ids(vecs, int_ids)
        self.save_path = self.save_path + self.request + ".faiss"
        faiss.write_index(index_raw, self.save_path)

class FaissQuery:
    '''
    input:
    vector: Embedding vector of documents List[List[float]] [MUST BE NORMALIZED]
    top_k: Number of nearest neighbors to retrieve int
    faiss_path: Path to the saved index file str
    output:
    doc_id: doc_id of documents List[str]
    '''
    def __init__(self, vector, top_k, faiss_path):
        self.vector = vector
        self.top_k = top_k
        self.faiss_path = faiss_path

    def query(self):
        # 读取索引文件
        index = faiss.read_index(self.faiss_path)
        
        # 将查询向量转换为numpy数组
        query_vecs = np.array(self.vector, dtype=np.float32)
        
        # 执行向量搜索，返回最近邻的索引和距离
        distances, indices = index.search(query_vecs, self.top_k)
        
        # 由于我们使用的是IndexFlatIP（内积），距离值就是相似度分数
        scores = distances.tolist()
        
        # 将索引转换回字符串ID
        doc_ids = [str(idx) for idx in indices[0]]
        
        # 将分数和文档ID组合并排序
        results = list(zip(scores[0], doc_ids))
        results.sort(reverse=True)  # 按分数降序排序
        
        # 分离排序后的结果
        sorted_scores = [score for score, _ in results]
        sorted_doc_ids = [doc_id for _, doc_id in results]
        
        return sorted_doc_ids, sorted_scores
    

'''
if __name__ == "__main__":
    id = ["1", "2", "3"]
    vector = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
    request = "test"
    save_path = "./test/"
    
    # 规范化向量用于测试
    vector_np = np.array(vector)
    norms = np.linalg.norm(vector_np, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized_vector = (vector_np / norms).tolist()
    
    # 保存索引
    faiss_saver = FaissSaver(id, normalized_vector, request, save_path)
    index_path = faiss_saver.save()
    
    # 查询测试
    query_vector = [[0.1, 0.2, 0.3]]  # 将进行内部规范化
    faiss_query = FaissQuery(query_vector, request, save_path)
    doc_ids, scores = faiss_query.query()
    print("查询结果文档ID:", doc_ids)
    print("相似度分数:", scores)
'''
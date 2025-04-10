
from dpr.vectorizer import query_vectorizer as qv
from faiss_composer.base import FaissQuery as fq
class query2doclist:
    '''
    input: query str, api_key str, model_name str, faiss_path str. top_k int
    output: doc_id List[str]
    '''
    def __init__(self, query: str, api_key: str, model_name: str, faiss_path: str, top_k: int):
        self.query = query
        self.api_key = api_key
        self.model_name = model_name
        self.vectorizer = qv(query, api_key, model_name)
        self.faiss_path = faiss_path
        self.top_k = top_k
    def query2doclist(self):
        query_vector = self.vectorizer.vectorize()
        faiss_query = fq([query_vector], self.top_k, self.faiss_path)
        doc_id_list, scores = faiss_query.query()
        sorted_doc_id_list = [doc_id_list[i] for i in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)]
        dict = {"document_id": sorted_doc_id_list, "scores": scores}
        return dict
    
    


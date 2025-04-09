import json
class doc_jsonler:
    '''
    input: json_path
    output: doc_ids List[str], contexts List[List[str]]
    '''
    def __init__(self, json_path):
        self.json_path = json_path
    
    def get_json_data(self):
        doc_ids = []
        contexts = []
        
        with open(self.json_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                doc_ids.append(data.get('doc_id', ''))
                contexts.append(data.get('text', []))
        
        return doc_ids, contexts

class query_jsonler:
    '''
    input: json_path
    output: queries List[str]
    '''
    def __init__(self, json_path):
        self.json_path = json_path

    def get_json_data(self):
        queries = []
        with open(self.json_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                queries.append(data.get('question', ''))
        return queries
'''
if __name__ == "__main__":
    # 测试文档读取
    doc_path = "./processed_plain.jsonl"
    doc_jsonler = doc_jsonler(doc_path)
    doc_ids, contexts = doc_jsonler.get_json_data()
    print(doc_ids[0])
    print(contexts[0])
    
    # 测试查询读取
    query_path = "./val.jsonl"
    query_jsonler = query_jsonler(query_path)
    queries = query_jsonler.get_json_data()
    print(queries[0])
'''
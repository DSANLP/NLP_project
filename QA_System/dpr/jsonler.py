import json

class doc_jsonler:
    '''
    input: json_path
    output: doc_ids, contexts
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
    output: queries
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
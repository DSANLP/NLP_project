class calculate_score:
    '''
    input: query str, txt_document_id_list list[str], txt_score_list list[float], dpr_document_id_list list[str], dpr_score_list list[float]
    output: final score
    '''
    def __init__(self, query, txt_document_id_list, dpr_document_id_list):
        self.query = query
        self.txt_document_id_list = txt_document_id_list
        self.dpr_document_id_list = dpr_document_id_list
    
    def calculate_score(self):
        # 合并两个文档ID列表并去重
        combined_doc_ids = list(set(self.txt_document_id_list + self.dpr_document_id_list))
        return self.query, combined_doc_ids
    
    

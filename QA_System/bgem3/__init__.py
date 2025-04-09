"""
BGE-M3向量化工具模块
包含文档向量化和高效处理的功能
"""

# 导入核心类
from dpr.vectorizer import doc_vectorizer
from dpr.jsonler import doc_jsonler, query_jsonler

# 为便于使用，从dpr模块导入并重新导出
__all__ = ['doc_vectorizer', 'doc_jsonler', 'query_jsonler'] 
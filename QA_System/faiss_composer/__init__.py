"""
FAISS向量检索模块
提供高效的向量存储和检索功能
"""

# 导入核心类
from faiss_composer.base import FaissSaver, FaissQuery

# 为便于使用，导出主要类
__all__ = ['FaissSaver', 'FaissQuery']

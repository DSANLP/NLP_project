import os
import time
import yaml
from utils import info, debug, warning, error, critical

class Compass:
    """嵌入方法选择器，用于在训练/向量化模式下选择不同的嵌入方法"""
    
    def __init__(self, config_path="config.yaml"):
        """
        初始化嵌入方法选择器
        
        参数:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        info(f"[Compass] 初始化嵌入方法选择器，配置文件路径: {config_path}")
        
        start_time = time.time()
        self.config = self.load_config(config_path)
        self.retrieval_config = self.config.get("retrieval", {})
        self.eval_config = self.config.get("evaluation", {})
        
        debug(f"[Compass] 配置加载完成，耗时: {time.time() - start_time:.2f}秒")
        info(f"[Compass] 检索配置: {', '.join([f'{k}={v}' for k, v in self.retrieval_config.items() if not isinstance(v, dict)])}")
    
    def load_config(self, config_path):
        """加载配置文件"""
        debug(f"[Compass] 开始加载配置文件: {config_path}")
        if not os.path.exists(config_path):
            error_msg = f"配置文件不存在: {config_path}"
            error(f"[Compass] {error_msg}")
            raise FileNotFoundError(error_msg)
            
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            debug(f"[Compass] 配置文件加载成功: {config_path}")
            return config
        except Exception as e:
            error_msg = f"加载配置文件失败: {str(e)}"
            error(f"[Compass] {error_msg}")
            raise
    
    def choose_embedding_method(self):
        """提示用户选择嵌入方法"""
        
        debug(f"[Compass] 选择Embedding方法")
        
        # 提示用户选择方法
        info(f"[Compass] 显示Embedding方法选择菜单")
        print("\n" + "="*50)
        print("请选择Embedding方法:")
        print("[1] API Embedding (BGE-M3)")
        print("[2] Local Embedding (DPR预训练模型)")
        print("[3] Train DPR模型")
        print("="*50)
        
        choice = input("请输入选项编号 [1/2/3]: ")
        debug(f"[Compass] 用户输入选项: {choice}")
        
        if choice == "1":
            method = "bge_m3"
        elif choice == "2":
            method = "dpr"
        elif choice == "3":
            method = "train_dpr"
        else:
            warning(f"[Compass] 无效选项: {choice}，请重新选择")
            return self.choose_embedding_method()
        
        info(f"[Compass] 选择的Embedding方法: {method}")
        return method
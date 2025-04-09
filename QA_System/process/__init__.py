from .process import DocumentProcessor, PlainTextProcessor, DocumentIndexer, DocumentPipeline

__all__ = ['DocumentProcessor', 'PlainTextProcessor', 'DocumentIndexer', 'DocumentPipeline', 'process_documents']

def process_documents(config):
    """
    处理文档的主函数
    
    参数:
        config: 配置对象
    """
    # 修正文件路径，确保路径存在且正确
    # 根据配置获取文件路径，确保路径以 / 结尾
    input_dir = config.get("data", {}).get("input_file", "./data/origin_data/")
    if not input_dir.endswith("/"):
        input_dir += "/"
    
    # 使用绝对路径并确保目录存在
    import os
    input_path = os.path.abspath(os.path.join(input_dir, "documents.jsonl"))
    
    # 输出目录路径处理
    model_output_dir = config.get("data", {}).get("model_output_dir", "./model/pkl")
    if not model_output_dir.endswith("/"):
        model_output_dir += "/"
    model_output_dir = os.path.abspath(model_output_dir)
    
    data_output_dir = config.get("data", {}).get("data_output_dir", "./data/processed_data")
    if not data_output_dir.endswith("/"):
        data_output_dir += "/"
    data_output_dir = os.path.abspath(data_output_dir)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(model_output_dir), exist_ok=True)
    os.makedirs(os.path.dirname(data_output_dir), exist_ok=True)
    
    # 导入日志函数
    from utils import log_message
    
    # 打印路径信息
    log_message(f"输入文件路径: {input_path}")
    log_message(f"模型输出目录: {model_output_dir}")
    log_message(f"数据输出目录: {data_output_dir}")
    
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        error_msg = f"错误：输入文件 {input_path} 不存在"
        log_message(error_msg)
        print(error_msg)
        import sys
        sys.exit(1)
    
    log_message("成功导入DocumentPipeline...")
    
    # 创建DocumentPipeline实例
    pipeline = DocumentPipeline(
        input_path=input_path, 
        model_output_dir=model_output_dir, 
        data_output_dir=data_output_dir
    )
    log_message(f"成功初始化DocumentPipeline...，当前参数为：input_path={input_path}, model_output_dir={model_output_dir}, data_output_dir={data_output_dir}")
    
    # 调用实例的run方法
    pipeline.run()
    
    log_message(f"常规统计方法模型训练完毕，模型已保存至：{model_output_dir}，处理好的数据保存到：{data_output_dir}")
    
    return {
        "input_path": input_path,
        "model_output_dir": model_output_dir,
        "data_output_dir": data_output_dir
    }


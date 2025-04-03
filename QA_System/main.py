import argparse
import os
import sys
import yaml
from webui.base import create_webui

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="问答系统")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    config_path = args.config
    
    # 检查配置文件是否存在
    if not os.path.exists(config_path):
        print(f"错误：配置文件 {config_path} 不存在")
        sys.exit(1)
    
    # 加载配置
    config = load_config(config_path)
    
    # 检查训练和评估模式
    train_mode = config.get("debug", {}).get("train_mode", False)
    evaluate_mode = config.get("debug", {}).get("evaluate_mode", False)
    
    if train_mode:
        print("系统运行在训练模式下，若需要启动webui，请将train_mode设置为false...")
        # 这里可以添加训练代码或调用训练模块
        pass
    elif evaluate_mode:
        print("系统运行在评估模式下，若需要启动webui，请将evaluate_mode设置为false...")
        # 这里可以添加评估代码或调用评估模块
        pass
    else:
        # 只有在非训练和非评估模式下才启动WebUI
        print("系统运行在常规模式下，正在启动WebUI...")
        create_webui(config_path)

if __name__ == "__main__":
    main()

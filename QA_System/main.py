import sys
import yaml
import os
import argparse
from utils import initialize_logger, log_message, set_log_level, LOG_LEVEL_DEBUG, LOG_LEVEL_INFO, LOG_LEVEL_WARNING, debug, error

def check_dependencies():
    """检查必要的依赖是否已安装"""
    try:
        # 必要的依赖库列表
        required_packages = ["gradio", "nltk", "numpy", "sklearn", "tqdm", "rank_bm25"]
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            log_message(f"缺少以下必要的依赖库: {', '.join(missing_packages)}")
            log_message("请使用 pip install -r requirements.txt 安装所有依赖")
            return False
        
        return True
    except Exception as e:
        log_message(f"检查依赖时出错: {str(e)}")
        return False

def load_config(config_path="config.yaml"):
    """加载配置文件"""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"加载配置文件出错: {str(e)}")
        sys.exit(1)

def main():
    """主函数，根据配置运行相应的模式"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="问答系统")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--verbose", action="store_true", help="启用详细日志")
    parser.add_argument("--quiet", action="store_true", help="只显示警告和错误")
    args = parser.parse_args()
    
    # 初始化日志记录器
    initialize_logger()
    
    # 设置日志级别
    if args.verbose:
        set_log_level(LOG_LEVEL_DEBUG)
        log_message("已启用详细日志模式")
    elif args.quiet:
        set_log_level(LOG_LEVEL_WARNING)
        log_message("已启用静默日志模式")
    else:
        set_log_level(LOG_LEVEL_INFO)
    
    # 检查依赖
    if not check_dependencies():
        log_message("缺少必要的依赖库，程序无法正常运行")
        sys.exit(1)
    
    # 加载配置
    config = load_config(args.config)
    
    # 获取运行模式（优先从debug部分获取配置）
    debug_config = config.get("debug", {})
    process_mode = debug_config.get("process_mode", False)
    evaluate_mode = debug_config.get("evaluate_mode", False)
    train_mode = debug_config.get("train_mode", False)
    
    # 输出详细的配置信息
    log_message(f"配置文件路径: {args.config}")
    debug(f"完整配置: {config}")
    log_message(f"运行模式: process_mode={process_mode}, evaluate_mode={evaluate_mode}, train_mode={train_mode}")
    
    # 根据模式运行对应功能
    if train_mode:
        log_message("启动训练模式...")
        try:
            from bert_base.train import run_dpr_training
            
            # 运行DPR训练
            success = run_dpr_training(args.config)
            
            if success:
                log_message("DPR模型训练完成!")
            else:
                log_message("DPR模型训练失败，请检查日志获取详细信息")
                sys.exit(1)
        except ImportError as e:
            log_message(f"导入训练模块时出错: {str(e)}")
            log_message("请确保安装了所有必要的依赖库: transformers, torch, faiss-cpu")
            sys.exit(1)
        except Exception as e:
            log_message(f"训练模式出错: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    elif process_mode:
        log_message("启动处理模式...")
        try:
            from process.process import DocumentPipeline
            
            # 获取文件路径
            input_dir = config.get("data", {}).get("input_file", "./data/origin_data/")
            if not input_dir.endswith("/"):
                input_dir += "/"
            
            input_path = os.path.abspath(os.path.join(input_dir, "documents.jsonl"))
            
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
            
            # 打印路径信息
            log_message(f"输入文件路径: {input_path}")
            log_message(f"模型输出目录: {model_output_dir}")
            log_message(f"数据输出目录: {data_output_dir}")
            
            # 创建DocumentPipeline实例
            pipeline = DocumentPipeline(
                input_path=input_path, 
                model_output_dir=model_output_dir, 
                data_output_dir=data_output_dir
            )
            
            # 调用实例的run方法
            pipeline.run()
            
        except Exception as e:
            log_message(f"处理模式出错: {str(e)}")
            print(f"处理模式出错: {str(e)}")
            raise
    elif evaluate_mode:
        log_message("启动评估模式...")
        from eval.eval import Compass
        compass = Compass(args.config)
        compass.run_evaluation()
    else:
        log_message("启动Web UI模式...")
        try:
            from webui.base import create_webui
            log_message("成功导入WebUI模块...")
            
            # 设置环境变量以避免潜在的线程问题
            os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            
            # 启动WebUI
            log_message("正在启动WebUI，这可能需要一些时间...")
            success = create_webui(args.config)
            
            if not success:
                log_message("WebUI启动失败，请检查日志获取详细信息")
                sys.exit(1)
            
        except KeyboardInterrupt:
            log_message("用户中断，退出程序")
            sys.exit(0)
        except Exception as e:
            error(f"启动WebUI时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
if __name__ == "__main__":
    main()

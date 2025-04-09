import sys
import yaml
import os
import argparse
from utils import initialize_logger, log_message, set_log_level, LOG_LEVEL_DEBUG, LOG_LEVEL_INFO, LOG_LEVEL_WARNING, debug, error, info, warning

def check_dependencies():
    """检查必要的依赖是否已安装"""
    try:
        # 必要的依赖库列表
        required_packages = ["gradio", "nltk", "numpy", "sklearn", "tqdm", "rank_bm25", "faiss"]
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

def vectorize_with_bgem3(config):
    """使用BGE-M3进行文档向量化并保存为FAISS索引"""
    try:
        info("开始BGE-M3文档向量化与FAISS索引建立流程...")
        
        # 导入必要的模块
        import bgem3
        import faiss
        import numpy as np
        import json
        import time
        import traceback
        from pathlib import Path
        
        # 获取向量化配置
        bgem3_config = config.get("retrieval", {}).get("deep_retrieval", {}).get("bgem3", {})
        faiss_config = config.get("retrieval", {}).get("faiss", {})
        
        # 解析配置
        model_name = bgem3_config.get("model", "BAAI/bge-m3")
        api_key = bgem3_config.get("api_key", "")
        doc_path = bgem3_config.get("doc_path", "./data/processed_data/processed_plain.jsonl")
        intermediate_dir = bgem3_config.get("intermediate_path", "./cache/bge_m3_intermediate")
        chunk_size = bgem3_config.get("chunk_size", 10)
        overlap_ratio = max(0.0, min(0.5, bgem3_config.get("overlap_ratio", 0.2)))
        use_weighted_avg = bgem3_config.get("use_weighted_avg", True)
        auto_load = bgem3_config.get("auto_load", True)
        
        # 获取FAISS配置
        index_name = faiss_config.get("index_name", "bgem3")
        index_path = faiss_config.get("index_path", "./faiss/")
        
        # 输出配置信息
        info(f"向量化配置:")
        info(f"- 文档路径: {doc_path}")
        info(f"- 模型: {model_name}")
        info(f"- 重叠率: {overlap_ratio:.2f}")
        info(f"- 批处理大小: {chunk_size}")
        info(f"- 使用加权平均: {use_weighted_avg}")
        info(f"- 中间结果目录: {intermediate_dir}")
        info(f"- 自动加载中间结果: {auto_load}")
        info(f"- FAISS索引路径: {index_path}")
        info(f"- 索引名称: {index_name}")
        
        # 定义辅助函数
        def save_intermediate_results(doc_ids, vectors, save_path=intermediate_dir):
            """保存中间结果到文件"""
            os.makedirs(save_path, exist_ok=True)
            timestamp = int(time.time())
            result_path = f"{save_path}/vectors_{timestamp}.json"
            
            # 将numpy数组转换为列表
            if isinstance(vectors, np.ndarray):
                vectors = vectors.tolist()
            
            data = {
                "doc_ids": doc_ids,
                "vectors": vectors
            }
            
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(data, f)
            
            info(f"已保存中间结果到 {result_path}，包含 {len(doc_ids)} 个文档向量")
            return result_path

        def load_intermediate_results(result_path):
            """从文件加载中间结果"""
            with open(result_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return data["doc_ids"], data["vectors"]

        def get_latest_intermediate_file(intermediate_path):
            """获取最新的中间结果文件"""
            if not os.path.exists(intermediate_path):
                return None
                
            files = [f for f in os.listdir(intermediate_path) if f.startswith("vectors_") and f.endswith(".json")]
            if not files:
                return None
                
            # 按时间戳排序找出最新的文件
            latest_file = sorted(files, key=lambda x: int(x.split("_")[1].split(".")[0]), reverse=True)[0]
            return f"{intermediate_path}/{latest_file}"
        
        # 创建必要的目录
        os.makedirs(intermediate_dir, exist_ok=True)
        os.makedirs(index_path, exist_ok=True)
        
        # 尝试加载之前的中间结果
        latest_file = get_latest_intermediate_file(intermediate_dir)
        all_doc_ids = []
        all_vectors = []
        start_index = 0
        
        if latest_file and auto_load:
            info(f"自动加载最新的中间结果: {latest_file}")
            all_doc_ids, all_vectors = load_intermediate_results(latest_file)
            info(f"已加载 {len(all_doc_ids)} 个文档向量")
        elif latest_file:
            load_choice = input(f"发现之前的中间结果: {latest_file}，是否加载? (y/n): ").strip().lower()
            if load_choice == 'y':
                all_doc_ids, all_vectors = load_intermediate_results(latest_file)
                info(f"已加载 {len(all_doc_ids)} 个文档向量")
        
        # 加载文档数据
        info("开始加载文档数据...")
        if not os.path.exists(doc_path):
            error(f"文档文件不存在: {doc_path}")
            return False
            
        doc_jsonler = bgem3.doc_jsonler(doc_path)
        doc_ids, contexts = doc_jsonler.get_json_data()
        info(f"成功加载 {len(doc_ids)} 个文档")
        
        # 计算总文档数和预估处理成本
        total_docs = len(doc_ids)
        
        # 创建一个向量化器实例用于估算处理成本和时间
        temp_vectorizer = bgem3.doc_vectorizer(doc_ids, contexts, api_key, model_name)
        estimation = temp_vectorizer.estimate_processing_cost(sample_size=min(100, total_docs))
        
        info("===== 处理估算 =====")
        info(f"总文档数: {estimation['total_documents']}")
        info(f"样本大小: {estimation['sample_size']}")
        info(f"平均每文档token数: {estimation['avg_tokens_per_doc']:.1f}")
        info(f"预估总token数: {estimation['estimated_total_tokens']:.0f}")
        info(f"预估处理时间: {estimation['estimated_time']['formatted']} (HH:MM:SS)")
        info(f"预估API成本: ¥{estimation['estimated_cost_cny']:.2f} CNY")
        info("===================")
        
        # 确认是否继续
        if not auto_load and total_docs > 10:
            confirm = input("是否继续处理? (y/n): ").strip().lower()
            if confirm != 'y':
                info("已取消处理")
                return False
        
        # 如果已加载中间结果，确定开始处理的索引位置
        if all_doc_ids:
            # 查找已处理的最后一个文档在原始文档列表中的位置
            processed_doc_ids_set = set(all_doc_ids)
            for i in range(len(doc_ids)):
                if doc_ids[i] not in processed_doc_ids_set:
                    start_index = i
                    break
            info(f"继续从第 {start_index+1} 个文档开始处理（跳过 {start_index} 个已处理的文档）")
        
        # 开始处理文档
        processed_count = len(all_doc_ids)
        start_time = time.time()
        
        try:
            for i in range(start_index, total_docs, chunk_size):
                try:
                    end_idx = min(i + chunk_size, total_docs)
                    info(f"处理文档 {i+1} 到 {end_idx} (共 {total_docs} 个，已完成 {((i+processed_count-start_index)/total_docs*100):.1f}%)...")
                    
                    # 获取当前批次的文档
                    batch_doc_ids = doc_ids[i:end_idx]
                    batch_contexts = contexts[i:end_idx]
                    
                    # 创建批次向量化器
                    batch_vectorizer = bgem3.doc_vectorizer(
                        batch_doc_ids, 
                        batch_contexts, 
                        api_key, 
                        model_name
                    )
                    
                    # 处理当前批次，使用带重叠的增强处理方法
                    result_ids, result_vectors = batch_vectorizer.process_documents_enhanced(
                        chunk_size=1, 
                        overlap_ratio=overlap_ratio,
                        use_weighted_avg=use_weighted_avg
                    )
                    
                    # 添加到结果中
                    all_doc_ids.extend(result_ids)
                    all_vectors.extend(result_vectors)
                    
                    processed_count += len(result_ids)
                    elapsed_time = time.time() - start_time
                    docs_per_second = (processed_count - len(all_doc_ids) + (i-start_index)) / elapsed_time if elapsed_time > 0 else 0
                    
                    info(f"已完成 {len(all_doc_ids)}/{total_docs} 个文档的处理 (速度: {docs_per_second:.2f} 文档/秒)")
                    
                    # 定期保存中间结果
                    if i > start_index and (len(all_doc_ids) % 50 == 0 or end_idx == total_docs):
                        save_intermediate_results(all_doc_ids, all_vectors, intermediate_dir)
                    
                    # 动态调整等待时间，根据文档处理速度
                    if end_idx < total_docs:
                        # 如果处理速度快，减少等待时间；如果处理速度慢，增加等待时间
                        if docs_per_second > 0.5:  # 每2秒以上处理一个文档
                            wait_time = 1
                        else:
                            wait_time = 3
                        debug(f"等待 {wait_time} 秒后处理下一批次...")
                        time.sleep(wait_time)
                    
                except Exception as e:
                    # 保存已处理的结果
                    if all_doc_ids:
                        save_path = save_intermediate_results(all_doc_ids, all_vectors, intermediate_dir)
                        warning(f"处理过程中出错，已保存中间结果到 {save_path}")
                    error(f"错误详情: {str(e)}")
                    traceback.print_exc()
                    break
            
            # 处理完成后，保存到FAISS索引
            if all_doc_ids:
                info("正在保存到FAISS索引...")
                
                # 确保向量已规范化
                vectors = np.array(all_vectors)
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                zero_mask = norms == 0
                norms[zero_mask] = 1.0  # 避免除以零
                normalized_vectors = vectors / norms
                
                # 使用FaissSaver保存索引
                faiss_saver = faiss.FaissSaver(all_doc_ids, normalized_vectors.tolist(), index_name, index_path)
                index_path_full = faiss_saver.save()
                info(f"索引已保存到: {index_path_full}")
                
                # 查询测试功能
                test_query = input("是否进行查询测试? (y/n): ").strip().lower()
                if test_query == 'y':
                    # 导入所需模块
                    try:
                        from dpr.vectorizer import query_vectorizer
                        
                        while True:
                            # 提供两种测试方式
                            print("\n查询测试选项:")
                            print("[1] 使用示例向量进行测试")
                            print("[2] 输入文本进行测试")
                            print("[0] 退出测试")
                            
                            choice = input("请选择测试方式 [0-2]: ")
                            
                            if choice == "0":
                                break
                            elif choice == "1":
                                # 使用第一个文档向量进行测试
                                query_vector = [normalized_vectors[0].tolist()]
                                faiss_query = faiss.FaissQuery(
                                    query_vector,
                                    index_name,
                                    index_path,
                                    k=5
                                )
                                try:
                                    result_doc_ids, scores = faiss_query.query()
                                    print("\n示例向量查询结果:")
                                    for i, (doc_id, score) in enumerate(zip(result_doc_ids, scores)):
                                        print(f"[{i+1}] 文档ID: {doc_id}, 相似度分数: {score:.4f}")
                                except Exception as e:
                                    error(f"查询测试失败: {str(e)}")
                                    
                            elif choice == "2":
                                # 输入文本进行测试
                                query_text = input("请输入查询文本: ")
                                if query_text.strip():
                                    # 使用query_vectorizer生成查询向量
                                    vectorizer = query_vectorizer([query_text], api_key, model_name)
                                    print("正在生成查询向量...")
                                    query_vectors = vectorizer.vectorize()
                                    
                                    # 执行查询
                                    faiss_query = faiss.FaissQuery(
                                        query_vectors,
                                        index_name,
                                        index_path,
                                        k=5
                                    )
                                    try:
                                        result_doc_ids, scores = faiss_query.query()
                                        print("\n文本查询结果:")
                                        for i, (doc_id, score) in enumerate(zip(result_doc_ids, scores)):
                                            # 获取原始文档内容
                                            try:
                                                doc_index = doc_ids.index(doc_id)
                                                doc_content = contexts[doc_index]
                                                # 截取前100个字符作为预览
                                                preview = doc_content[:100] + "..." if len(doc_content) > 100 else doc_content
                                                print(f"[{i+1}] 文档ID: {doc_id}, 相似度: {score:.4f}")
                                                print(f"    预览: {preview}\n")
                                            except:
                                                print(f"[{i+1}] 文档ID: {doc_id}, 相似度: {score:.4f}")
                                    except Exception as e:
                                        error(f"查询测试失败: {str(e)}")
                                        traceback.print_exc()
                            else:
                                print("无效选项，请重新选择")
                    except Exception as e:
                        error(f"初始化查询测试环境失败: {str(e)}")
                        traceback.print_exc()
        
        except KeyboardInterrupt:
            info("检测到用户中断，正在保存当前结果...")
            if all_doc_ids:
                save_path = save_intermediate_results(all_doc_ids, all_vectors, intermediate_dir)
                info(f"中间结果已保存到: {save_path}")
        
        info("向量化与FAISS索引建立流程完成！")
        return True
        
    except Exception as e:
        error(f"向量化过程出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数，根据配置运行相应的模式"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="问答系统")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--verbose", action="store_true", help="启用详细日志")
    parser.add_argument("--quiet", action="store_true", help="只显示警告和错误")
    parser.add_argument("--mode", type=str, choices=["process", "train", "evaluate", "webui", "vectorize"],
                       help="运行模式: process=数据处理, train=模型训练, evaluate=模型评估, webui=启动网页界面, vectorize=文档向量化")
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
    
    # 获取运行模式（优先从命令行参数获取，其次从debug部分获取配置）
    run_mode = args.mode if args.mode else None
    
    # 如果命令行未指定模式，则从配置文件获取
    if not run_mode:
        debug_config = config.get("debug", {})
        process_mode = debug_config.get("process_mode", False)
        evaluate_mode = debug_config.get("evaluate_mode", False)
        train_mode = debug_config.get("train_mode", False)
        vectorize_mode = debug_config.get("vectorize_mode", False)
        rerank_mode = debug_config.get("rerank_mode", False)
    else:
        # 根据命令行参数设置模式
        process_mode = (run_mode == "process")
        evaluate_mode = (run_mode == "evaluate")
        train_mode = (run_mode == "train")
        vectorize_mode = (run_mode == "vectorize")
        rerank_mode = (run_mode == "rerank")
    # 输出详细的配置信息
    log_message(f"配置文件路径: {args.config}")
    debug(f"完整配置: {config}")
    log_message(f"运行模式: {'命令行指定: '+run_mode if run_mode else '配置文件指定'}")
    log_message(f"运行模式详情: process={process_mode}, evaluate={evaluate_mode}, train={train_mode}, vectorize={vectorize_mode}, rerank={rerank_mode}")
    
    if train_mode:
        log_message("启动训练模式...")
        
        # 获取当前的训练模块
        active_modules = config.get("modules", {}).get("active", {})
        retrieval_method = active_modules.get("retrieval", "hybrid")
        
        log_message(f"当前检索方法: {retrieval_method}")
        
        # 检查bert_base配置是否存在且非空
        bert_base_config = config.get("bert_base", {}).get("train", {})
        has_bert_base_config = bool(bert_base_config)
        
        # 检查debug.train_mode是否明确指定为True
        explicitly_train_bert = config.get("debug", {}).get("train_mode", False)
        
        # 优先训练bert_base（如果配置存在且明确指定了训练模式）
        if has_bert_base_config and explicitly_train_bert:
            # 导入BERT-Base训练模块
            log_message("准备训练BERT-Base模型...")
            try:
                from bert_base.train import train_bert_base_model
                train_bert_base_model(config)
            except Exception as e:
                error(f"BERT-Base模型训练出错: {str(e)}")
                import traceback
                traceback.print_exc()
        elif retrieval_method == "bert_base":
            # 如果retrieval方法指定为bert_base，也训练bert_base
            log_message("准备训练BERT-Base模型...")
            try:
                from bert_base.train import train_bert_base_model
                train_bert_base_model(config)
            except Exception as e:
                error(f"BERT-Base模型训练出错: {str(e)}")
                import traceback
                traceback.print_exc()
        elif retrieval_method == "hybrid" or retrieval_method == "bm25":
            # 其他情况下，继续使用Compass选择嵌入方法
            try:
                from dpr.compass import Compass
                # 创建Compass实例，传递完整的配置对象而不是配置文件路径
                compass = Compass(config_path=args.config)
                method = compass.choose_embedding_method()
                if method == "bge_m3":
                    log_message("BGE-M3不需要训练，请启用vectorize_mode进行向量化")
                elif method == "dpr":
                    log_message("使用预训练的DPR模型，无需训练")
                elif method == "train_dpr":
                    log_message("DPR模型训练暂未实现")
            except Exception as e:
                error(f"选择嵌入方法出错: {str(e)}")
                import traceback
                traceback.print_exc()
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
                data_output_dir=data_output_dir,
                config=config  # 传递完整配置
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
    elif vectorize_mode:
        log_message("启动文档向量化模式...")
        vectorize_with_bgem3(config)
    elif rerank_mode:
        log_message("启动重排序模式...")
        from eval.combine import calculate_score
        from bm25.base import ManualQuerySearch
        from dpr.jsonler import query_jsonler
        from webui.base import WebUI
        
        # 从配置文件或参数中获取验证集路径
        test_path = config.get("data", {}).get("test_path", "./data/original_data/test.jsonl")
        log_message(f"加载查询数据: {test_path}")
        
        # 使用query_jsonler加载所有查询
        query_loader = query_jsonler(test_path)
        queries = query_loader.get_json_data()
        log_message(f"成功加载 {len(queries)} 个查询")
        
        for query in queries:
            txt_doc_id = ManualQuerySearch(query).search()["document_id"]
            dpr_doc_id = None
            query, all_doc_id = calculate_score(query, txt_doc_id, dpr_doc_id)
            rerank_doc_id = WebUI().rerank_results(query, all_doc_id, top_n=5)
            doc_content = WebUI().retrieve_doc(rerank_doc_id)
            answer = WebUI().generate_answer(query, doc_content, n=5)
            result = {
                "question": query 
            }
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

import os
import sys
import yaml
import time
import json
from utils import info, debug, warning, error, critical

class Compass:
    """评估方向控制器，用于选择不同的评估方法"""
    
    def __init__(self, config_path="config.yaml"):
        """
        初始化评估控制器
        
        参数:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        info(f"[Compass] 初始化评估控制器，配置文件路径: {config_path}")
        
        start_time = time.time()
        self.config = self.load_config(config_path)
        self.eval_config = self.config.get("evaluation", {})
        self.retrieval_config = self.config.get("retrieval", {})
        
        debug(f"[Compass] 配置加载完成，耗时: {time.time() - start_time:.2f}秒")
        info(f"[Compass] 评估配置: {', '.join([f'{k}={v}' for k, v in self.eval_config.items() if not isinstance(v, dict)])}")
    
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
    
    def choose_eval_method(self):
        """提示用户选择评估方法或使用默认方法"""
        # 检查是否自动选择默认方法
        auto_select = self.eval_config.get("auto_select", False)
        default_method = self.eval_config.get("default_method", "text")
        
        debug(f"[Compass] 选择评估方法 - 自动选择: {auto_select}, 默认方法: {default_method}")
        
        if auto_select:
            info(f"[Compass] 自动选择评估方法: {default_method}")
            return default_method
        
        # 提示用户选择方法
        info(f"[Compass] 显示评估方法选择菜单")
        print("\n" + "="*50)
        print("请选择评估方法:")
        print("[1] 文本检索评估 (BM25/TF-IDF)")
        print("[2] 深度嵌入评估 (DPR)")
        print("[3] 混合检索评估 (Hybrid)")
        print("="*50)
        
        choice = input("请输入选项编号 [1/2/3]: ")
        debug(f"[Compass] 用户输入选项: {choice}")
        
        if choice == "1":
            method = "text"
        elif choice == "2":
            method = "deep_embedding"
        elif choice == "3":
            method = "hybrid"
        else:
            warning(f"[Compass] 无效选项: {choice}，使用默认方法: {default_method}")
            print(f"无效选项: {choice}，使用默认方法: {default_method}")
            method = default_method
        
        info(f"[Compass] 选择的评估方法: {method}")
        return method
    
    def get_retrieval_params(self, method_type):
        """
        获取特定检索方法的参数配置
        
        参数:
            method_type: 检索方法类型 (text/deep_embedding/hybrid)
            
        返回:
            检索方法参数字典
        """
        debug(f"[Compass] 获取{method_type}检索方法参数")
        
        # 获取通用检索参数
        params = {
            "top_k": self.retrieval_config.get("top_k", 5),
            "max_words": self.retrieval_config.get("max_words", 50),
            "use_plain_text": self.retrieval_config.get("use_plain_text", True)
        }
        
        # 根据检索类型获取特定参数
        if method_type == "text":
            text_config = self.retrieval_config.get("text_retrieval", {})
            params["method"] = text_config.get("method", "hybrid")
            params["hybrid_alpha"] = text_config.get("hybrid_alpha", 0.7)
            info(f"[Compass] 加载文本检索参数 - 方法: {params['method']}, 混合比例: {params['hybrid_alpha']}")
            
        elif method_type == "deep_embedding":
            from dpr import compass
            method = compass.choose_embedding_method()
            if method == "train_dpr":
                warning(f"[Evaluation] Evaluation mode is not supported in DPR training mode")
                sys.exit(1)
            elif method == "dpr":
                pass
            elif method == "bge_m3":
                deep_config = self.retrieval_config.get("deep_retrieval", {})
                m3_config = deep_config.get("api_embedding", {})
                params["model"] = m3_config.get("model", "BAAI/bge-m3")
                params["api_key"] = m3_config.get("api_key", "sk-proj-1234567890")
                params["doc_path"] = m3_config.get("doc_path", "./data/processed_data/processed_plain.jsonl")
                params["intermediate_path"] = m3_config.get("intermediate_path", "./cache/bge_m3_intermediate")
        elif method_type == "hybrid":
            print("需要进一步完善")
            pass
        
        debug(f"[Compass] 通用参数 - top_k: {params['top_k']}, max_words: {params['max_words']}, use_plain_text: {params['use_plain_text']}")
        return params
    
    def evaluate_text_retrieval(self):
        """评估文本检索方法 (BM25/TF-IDF)"""
        info(f"[Compass] 开始文本检索评估")
        start_time = time.time()
        
        # 获取文本检索参数
        text_params = self.get_retrieval_params("text")
        
        info(f"[Compass] 使用{text_params['method']}方法进行文本检索评估")
        
        # 处理文件路径
        data_output_dir = self.config.get("data", {}).get("data_output_dir", "./data/processed_data/")
        model_output_dir = self.config.get("data", {}).get("model_output_dir", "./model/pkl/")
        val_path = self.config.get("data", {}).get("val_path", "./data/original_data/val.jsonl")
        
        debug(f"[Compass] 路径配置 - 数据输出目录: {data_output_dir}")
        debug(f"[Compass] 路径配置 - 模型输出目录: {model_output_dir}")
        debug(f"[Compass] 路径配置 - 验证集路径: {val_path}")
        
        # 确保路径以 / 结尾
        if not data_output_dir.endswith("/"):
            data_output_dir += "/"
        if not model_output_dir.endswith("/"):
            model_output_dir += "/"
            
        # 转换为绝对路径
        data_output_dir = os.path.abspath(data_output_dir)
        model_output_dir = os.path.abspath(model_output_dir)
        val_path = os.path.abspath(val_path)
        
        debug(f"[Compass] 绝对路径 - 数据输出目录: {data_output_dir}")
        debug(f"[Compass] 绝对路径 - 模型输出目录: {model_output_dir}")
        debug(f"[Compass] 绝对路径 - 验证集路径: {val_path}")
        
        # 根据use_plain_text决定使用哪个处理后的文档
        use_plain_text = text_params.get("use_plain_text", True)
        doc_type = "plain" if use_plain_text else "markdown"
        doc_path = os.path.join(data_output_dir, f"processed_{doc_type}.jsonl")
        bm25_path = os.path.join(model_output_dir, f"{doc_type}_bm25.pkl")
        tfidf_path = os.path.join(model_output_dir, f"{doc_type}_tfidf.pkl")
        
        info(f"[Compass] 使用{'纯文本' if use_plain_text else 'Markdown'}格式文档")
        debug(f"[Compass] 文档路径: {doc_path}")
        debug(f"[Compass] BM25模型路径: {bm25_path}")
        debug(f"[Compass] TF-IDF模型路径: {tfidf_path}")
        
        # 检查必要文件是否存在
        required_files = [doc_path, bm25_path, tfidf_path, val_path]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            error(f"[Compass] 缺少必要文件:")
            for file in missing_files:
                error(f"[Compass] - 文件不存在: {file}")
            critical("[Compass] 请先运行处理模式(process_mode)生成必要的模型和数据文件")
            print("错误: 缺少必要文件，请查看日志了解详情")
            sys.exit(1)
        else:
            debug(f"[Compass] 所有必要文件检查通过")
        
        # 获取输出路径
        output_path = self.eval_config.get("output", {}).get("text", 
                          os.path.join(data_output_dir, f"text_evaluation_results.jsonl"))
        
        # 检查输出文件是否已存在
        if os.path.exists(output_path):
            info(f"[Compass] 发现已有评估结果: {output_path}")
            
            # 检查文件是否有内容
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                
                if first_line:
                    info(f"[Compass] 跳过评估过程，直接使用已有结果计算指标")
                    # 直接跳到指标计算部分
                    calculate_metrics = self.eval_config.get("metrics", {}).get("calculate", False)
                    if calculate_metrics:
                        self._calculate_metrics(output_path)
                    
                    info(f"[Compass] 文本检索评估流程完成")
                    return output_path
                else:
                    info(f"[Compass] 已有结果文件为空，将重新执行评估过程")
            except Exception as e:
                warning(f"[Compass] 检查已有结果文件时出错: {str(e)}，将重新执行评估过程")
        
        # 创建输出目录
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            debug(f"[Compass] 创建输出目录: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
        
        info(f"[Compass] 评估结果将保存至: {output_path}")
        
        # 构建评估器配置
        eval_config = {
            "method": text_params.get("method", "hybrid"),
            "hybrid_alpha": text_params.get("hybrid_alpha", 0.7),
            "top_k": text_params.get("top_k", 5),
            "max_words": text_params.get("max_words", 2),
            "doc_path": doc_path,
            "bm25_path": bm25_path,
            "tfidf_path": tfidf_path,
            "val_path": val_path,
            "output_path": output_path
        }
        
        debug(f"[Compass] 评估器配置准备完成")
        
        # 使用自定义的评估方法，直接生成符合metrics_calculation要求的格式
        try:
            info(f"[Compass] 开始执行评估过程")
            output_path = self._run_batch_evaluation_with_format(eval_config)
            info(f"[Compass] 评估完成，耗时: {time.time() - start_time:.2f}秒")
            info(f"[Compass] 评估结果保存至: {output_path}")
        except Exception as e:
            error(f"[Compass] 评估过程出错: {str(e)}")
            print(f"评估过程出错: {str(e)}")
            raise
        
        # 如果配置了计算评估指标
        calculate_metrics = self.eval_config.get("metrics", {}).get("calculate", False)
        if calculate_metrics:
            self._calculate_metrics(output_path)
        
        info(f"[Compass] 文本检索评估流程完成")
        return output_path
        
    def _run_batch_evaluation_with_format(self, config):
        """
        运行批量评估并直接输出符合metrics_calculation格式的结果
        
        参数:
            config: 评估配置
            
        返回:
            输出文件路径
        """
        debug(f"[Compass] 启动批量评估，使用符合metrics_calculation的输出格式")
        
        # 导入BatchEvaluator类但不直接使用其evaluate方法
        from bm25.base import BatchEvaluator, BaseSearchEngine
        
        # 创建自定义的评估器
        evaluator = BatchEvaluator(config)
        max_words = config.get("max_words", 2)
        
        # 加载文档
        doc_ids, docs = evaluator.load_documents(config["doc_path"])
        info(f"[Compass] 成功加载 {len(doc_ids)} 个文档")
        
        # 加载BM25和TF-IDF模型
        bm25 = evaluator.load_bm25(config["bm25_path"])
        tfidf_vectorizer, _ = evaluator.load_tfidf(config["tfidf_path"])
        tfidf_matrix = tfidf_vectorizer.transform(docs)
        
        # 加载验证集问题
        questions = evaluator.load_questions(config["val_path"])
        info(f"[Compass] 成功加载 {len(questions)} 个问题")
        
        # 存储评估结果（使用metrics_calculation所需的格式）
        results = []
        
        # 评估每个问题
        start_time = time.time()
        for i, question_item in enumerate(questions):
            # 每10%打印一次进度
            if (i+1) % max(1, len(questions)//10) == 0 or i+1 == len(questions):
                progress = (i+1) / len(questions) * 100
                elapsed = time.time() - start_time
                eta = (elapsed / (i+1)) * (len(questions) - i - 1) if i > 0 else 0
                info(f"[Compass] 进度: {progress:.1f}% ({i+1}/{len(questions)}), 已用时间: {elapsed:.1f}秒, 预计剩余: {eta:.1f}秒")
            
            # 从问题项目中获取查询文本
            query = question_item["question"]
            
            # 预测文档
            pred_doc_ids, _ = evaluator.predict_top_document(
                query, doc_ids, docs, tfidf_vectorizer, tfidf_matrix, bm25,
                method=config["method"], alpha=config["hybrid_alpha"], top_k=config["top_k"],
                return_format="lists"
            )
            
            # 如果有预测结果，从第一个文档提取答案；否则使用空字符串
            answer = ""
            if pred_doc_ids and len(pred_doc_ids) > 0:
                doc_idx = doc_ids.index(pred_doc_ids[0])
                doc_text = docs[doc_idx]
                answer = BaseSearchEngine.extract_answer_from_doc(doc_text, max_words=max_words)
            
            # 使用metrics_calculation期望的格式记录结果
            result = {
                "question": query,
                "answer": answer,
                "document_id": pred_doc_ids  # 列表类型
            }
            results.append(result)
        
        # 保存结果
        output_path = config["output_path"]
        with open(output_path, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        
        info(f"[Compass] 评估结果已保存，符合metrics_calculation格式")
        return output_path
    
    def evaluate_deep_embedding(self):
        """评估深度嵌入检索方法"""
        info(f"[Compass] 开始深度嵌入检索评估")
        start_time = time.time()

        # 从DPR模块导入Compass并选择嵌入方法
        from dpr.compass import Compass as DPRCompass
        method = DPRCompass(self.config_path).choose_embedding_method()
        
        if method == "train_dpr":
            warning(f"[Compass] 评估模式不支持DPR训练模式")
            print("评估模式不支持DPR训练模式，请选择其他嵌入方法。")
            sys.exit(1)
        
        # 获取嵌入检索参数
        embedding_params = self.get_retrieval_params("deep_embedding")
        
        # 处理文件路径
        data_output_dir = self.config.get("data", {}).get("data_output_dir", "./data/processed_data/")
        val_path = self.config.get("data", {}).get("val_path", "./data/original_data/val.jsonl")
        
        # 确保路径以 / 结尾
        if not data_output_dir.endswith("/"):
            data_output_dir += "/"
            
        # 转换为绝对路径
        data_output_dir = os.path.abspath(data_output_dir)
        val_path = os.path.abspath(val_path)
        
        debug(f"[Compass] 绝对路径 - 数据输出目录: {data_output_dir}")
        debug(f"[Compass] 绝对路径 - 验证集路径: {val_path}")
        
        output_path = None
        
        if method == "bge_m3":
            info(f"[Compass] 使用BGE-M3 API嵌入模型进行评估")
            
            # 获取BGE-M3专用参数
            model_name = embedding_params.get("model", "BAAI/bge-m3")
            api_key = embedding_params.get("api_key", "")
            doc_path = embedding_params.get("doc_path", "./data/processed_data/processed_plain.jsonl")
            intermediate_path = embedding_params.get("intermediate_path", "./cache/bge_m3_intermediate")
            top_k = embedding_params.get("top_k", 5)
            max_words = embedding_params.get("max_words", 50)
            
            # 创建输出目录
            output_dir = os.path.dirname(data_output_dir)
            if not os.path.exists(output_dir):
                debug(f"[Compass] 创建输出目录: {output_dir}")
                os.makedirs(output_dir, exist_ok=True)
            
            # 定义输出路径
            output_path = self.eval_config.get("output", {}).get("deep_embedding", 
                              os.path.join(data_output_dir, f"bge_m3_evaluation_results.jsonl"))
            
            info(f"[Compass] 评估结果将保存至: {output_path}")
            
            # 检查输出文件是否已存在
            if os.path.exists(output_path):
                info(f"[Compass] 发现已有评估结果: {output_path}")
                
                # 检查文件是否有内容
                try:
                    with open(output_path, 'r', encoding='utf-8') as f:
                        first_line = f.readline().strip()
                    
                    if first_line:
                        info(f"[Compass] 跳过评估过程，直接使用已有结果计算指标")
                        # 直接跳到指标计算部分
                        calculate_metrics = self.eval_config.get("metrics", {}).get("calculate", False)
                        if calculate_metrics:
                            self._calculate_metrics(output_path)
                        
                        info(f"[Compass] BGE-M3评估流程完成")
                        return output_path
                    else:
                        info(f"[Compass] 已有结果文件为空，将重新执行评估过程")
                except Exception as e:
                    warning(f"[Compass] 检查已有结果文件时出错: {str(e)}，将重新执行评估过程")
            
            # 导入必要的类
            from dpr.vectorizer import doc_vectorizer, query_vectorizer
            from dpr.jsonler import doc_jsonler, query_jsonler
            
            # 创建中间目录
            if not os.path.exists(intermediate_path):
                debug(f"[Compass] 创建中间文件目录: {intermediate_path}")
                os.makedirs(intermediate_path, exist_ok=True)
                
            # 加载文档
            info(f"[Compass] 加载文档: {doc_path}")
            doc_loader = doc_jsonler(doc_path)
            doc_ids, contexts = doc_loader.get_json_data()
            info(f"[Compass] 成功加载 {len(doc_ids)} 个文档")
            
            # 加载验证集问题
            info(f"[Compass] 加载验证集问题: {val_path}")
            query_loader = query_jsonler(val_path)
            queries = query_loader.get_json_data()
            query_data = []
            
            # 加载完整的验证集数据以获取参考文档ID
            with open(val_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        query_data.append(json.loads(line))
            
            info(f"[Compass] 成功加载 {len(queries)} 个问题")
            
            # 检查中间文件是否存在
            doc_vector_path = os.path.join(intermediate_path, "doc_vectors.json")
            doc_embedding_exists = os.path.exists(doc_vector_path)
            
            # 生成或加载文档向量
            if doc_embedding_exists:
                info(f"[Compass] 加载已有文档向量: {doc_vector_path}")
                with open(doc_vector_path, 'r', encoding='utf-8') as f:
                    doc_vectors_data = json.load(f)
                    doc_vectors = doc_vectors_data.get("vectors", [])
            else:
                info(f"[Compass] 生成文档向量，使用模型: {model_name}")
                doc_vec = doc_vectorizer(doc_ids, contexts, api_key, model_name)
                _, doc_vectors = doc_vec.vectorize()
                
                # 保存文档向量
                with open(doc_vector_path, 'w', encoding='utf-8') as f:
                    json.dump({"vectors": doc_vectors}, f)
                info(f"[Compass] 文档向量已保存至: {doc_vector_path}")
            
            # 生成查询向量并执行评估
            info(f"[Compass] 生成查询向量，使用模型: {model_name}")
            query_vec = query_vectorizer(queries, api_key, model_name)
            query_vectors = query_vec.vectorize()
            
            # 存储评估结果
            results = []
            import numpy as np
            from tqdm import tqdm
            
            # 将向量转换为numpy数组以加速计算
            doc_vectors_np = np.array(doc_vectors)
            
            # 评估每个查询
            info(f"[Compass] 开始评估，计算相似度并排序")
            for i, (query, query_vector) in enumerate(tqdm(zip(queries, query_vectors), total=len(queries))):
                # 使用点积计算相似度
                query_vector_np = np.array(query_vector)
                similarities = np.dot(doc_vectors_np, query_vector_np)
                
                # 获取相似度最高的top_k个文档
                top_indices = similarities.argsort()[-top_k:][::-1]
                pred_doc_ids = [doc_ids[idx] for idx in top_indices]
                
                # 提取答案
                answer = ""
                if pred_doc_ids and len(pred_doc_ids) > 0:
                    doc_text = contexts[doc_ids.index(pred_doc_ids[0])]
                    import re
                    # 简单截取前max_words个词作为答案
                    words = re.findall(r'\w+', doc_text)
                    answer = ' '.join(words[:max_words])
                
                # 记录结果
                result = {
                    "question": query,
                    "answer": answer,
                    "document_id": pred_doc_ids
                }
                results.append(result)
            
            # 保存结果
            info(f"[Compass] 保存评估结果: {output_path}")
            with open(output_path, "w", encoding="utf-8") as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
            
            info(f"[Compass] 评估完成，耗时: {time.time() - start_time:.2f}秒")
            
        elif method == "dpr":
            info(f"[Compass] DPR本地嵌入模型评估暂未实现")
            print("DPR本地嵌入模型评估功能尚未实现，请使用BGE-M3 API嵌入模型。")
            return None
        
        # 如果配置了计算评估指标且有输出结果
        if output_path:
            calculate_metrics = self.eval_config.get("metrics", {}).get("calculate", False)
            if calculate_metrics:
                self._calculate_metrics(output_path)
        
        info(f"[Compass] 深度嵌入检索评估流程完成")
        return output_path
    
    def evaluate_hybrid(self):
        """评估混合检索方法"""
        info(f"[Compass] 混合检索评估功能已被移除")
        warning(f"[Compass] 混合检索功能依赖于DPR，已被一并移除")
        print("混合检索评估功能已被移除，请联系系统管理员获取更多信息。")
        return None
    
    def run_evaluation(self):
        """运行评估流程"""
        info(f"[Compass] ====== 开始评估流程 ======")
        start_time = time.time()
        
        # 让用户选择评估方法
        eval_method = self.choose_eval_method()
        info(f"[Compass] 选择了评估方法: {eval_method}")
        
        # 根据选择的方法执行相应的评估
        result = None
        try:
            if eval_method == "text":
                result = self.evaluate_text_retrieval()
            elif eval_method == "deep_embedding":
                result = self.evaluate_deep_embedding()
            elif eval_method == "hybrid":
                result = self.evaluate_hybrid()
            else:
                warning(f"[Compass] 未知的评估方法: {eval_method}")
                print(f"未知的评估方法: {eval_method}")
                return None
        except Exception as e:
            error(f"[Compass] 评估流程出错: {str(e)}")
            print(f"评估流程出错: {str(e)}")
            raise
        finally:
            elapsed_time = time.time() - start_time
            info(f"[Compass] ====== 评估流程结束，总耗时: {elapsed_time:.2f}秒 ======")
        
        return result

    def _calculate_metrics(self, output_path):
        """
        计算并保存评估指标
        
        参数:
            output_path: 评估结果路径
            
        返回:
            metrics_results: 计算的指标结果字典
        """
        info(f"[Compass] 开始计算评估指标")
        
        try:
            # 导入评估指标计算模块
            from eval.metrics_calculation import calculate_metrics as calc_metrics
            
            # 获取验证集路径和预测结果路径
            val_path = self.config.get("data", {}).get("val_path", "./data/original_data/val.jsonl")
            
            # 获取指标类型
            metric_types = self.eval_config.get("metrics", {}).get("types", ["recall@5", "mrr@5"])
            info(f"[Compass] 计算以下评估指标: {', '.join(metric_types)}")
            
            info(f"[Compass] 使用验证集: {val_path}")
            info(f"[Compass] 使用预测结果: {output_path}")
            
            # 计算评估指标
            metrics_results = calc_metrics(val_path, output_path)
            
            # 只保留文档检索相关指标
            doc_metrics = {
                'recall@5': metrics_results['recall@5'],
                'mrr@5': metrics_results['mrr@5']
            }
            
            # 打印评估结果
            info(f"[Compass] 文档检索评估指标计算完成")
            for metric_name, value in doc_metrics.items():
                info(f"[Compass] {metric_name}: {value:.4f}")
            
            # 将评估结果保存到文件
            metrics_output_path = self.eval_config.get("metrics", {}).get("output_path", 
                              os.path.join(os.path.dirname(output_path), "retrieval_metrics.json"))
            
            with open(metrics_output_path, "w", encoding="utf-8") as f:
                json.dump(doc_metrics, f, ensure_ascii=False, indent=2)
            
            info(f"[Compass] 评估指标已保存到: {metrics_output_path}")
            return doc_metrics
            
        except Exception as e:
            error(f"[Compass] 计算评估指标时出错: {str(e)}")
            print(f"计算评估指标时出错: {str(e)}")
            return None


# 用于测试的主函数
if __name__ == "__main__":
    info("开始独立运行评估模块")
    try:
        compass = Compass()
        result = compass.run_evaluation()
        if result:
            print(f"评估完成，结果保存至: {result}")
        else:
            print("评估未完成或未生成结果")
    except Exception as e:
        error(f"评估过程发生错误: {str(e)}")
        print(f"评估过程发生错误: {str(e)}")
    info("评估模块运行结束")
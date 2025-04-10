import gradio as gr
import yaml
import os
from utils import log_message, debug, warning, error
import sys
import traceback
from backbone.llm import QwenChatClient
from backbone.rerank import ReRanker

class WebUI:
    def __init__(self, config_path="config.yaml"):
        """
        初始化WebUI类
        
        功能：
        - 加载配置文件
        - 设置WebUI相关参数（端口、是否展示界面）
        - 确定默认检索类型
        
        参数：
        - config_path: 配置文件路径，默认为"config.yaml"
        
        与用户界面联动：
        - 决定WebUI启动时使用的端口
        - 决定是否显示WebUI界面
        - 设置Radio按钮的默认选择值
        """
        self.config_path = config_path
        self.config = self.load_config(config_path)
        self.port = self.config.get("webui", {}).get("port", 8080)
        self.presentation = self.config.get("webui", {}).get("presentation", True)
        self.retrieval_type = self.config.get("modules", {}).get("active", {}).get("retrieval", "hybrid")
        
        # 初始化搜索器
        self.init_searcher()
    
    def init_searcher(self):
        """初始化BM25搜索器"""
        # 获取BM25配置
        bm25_config = self.config.get("retrieval", {}).get("text_retrieval", {})
        retrieval_method = bm25_config.get("method", "hybrid")
        
        # 处理文件路径
        data_output_dir = self.config.get("data", {}).get("data_output_dir", "./data/processed_data/")
        model_output_dir = self.config.get("data", {}).get("model_output_dir", "./model/pkl/")
        
        # 根据use_plain_text决定使用哪个处理后的文档
        use_plain_text = bm25_config.get("use_plain_text", True)
        doc_type = "plain" if use_plain_text else "markdown"
        doc_path = os.path.join(data_output_dir, f"processed_{doc_type}.jsonl")
        bm25_path = os.path.join(model_output_dir, f"{doc_type}_bm25.pkl")
        tfidf_path = os.path.join(model_output_dir, f"{doc_type}_tfidf.pkl")
        
        log_message(f"文档路径: {doc_path}")
        log_message(f"BM25模型路径: {bm25_path}")
        log_message(f"TF-IDF模型路径: {tfidf_path}")
        
        # 检查必要文件是否存在
        required_files = [doc_path, bm25_path, tfidf_path]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            for file in missing_files:
                log_message(f"错误：文件 {file} 不存在")
            log_message("请先运行处理模式(process_mode)生成必要的模型和数据文件")
            import sys
            sys.exit(1)
        
        # 构建查询搜索器配置
        self.search_config = {
            "method": retrieval_method,
            "hybrid_alpha": bm25_config.get("hybrid_alpha", 0.7),
            "top_k": bm25_config.get("top_k", 5),
            "max_words": bm25_config.get("max_words", 2),
            "doc_path": doc_path,
            "bm25_path": bm25_path,
            "tfidf_path": tfidf_path
        }
        
        try:
            # 导入并实例化搜索器
            from bm25.base import ManualQuerySearch
            log_message("成功导入ManualQuerySearch...")
            
            # 添加更详细的加载日志
            log_message("开始初始化搜索器...")
            log_message("正在加载文档，这可能需要一点时间...")
            log_message("正在加载BM25模型，这可能需要一点时间...")
            log_message("如果是hybrid模式，还将加载TF-IDF模型...")
            
            # 实例化搜索器
            self.searcher = ManualQuerySearch(self.search_config)
            log_message(f"成功初始化ManualQuerySearch，当前参数为：{self.search_config}")
            debug("搜索器初始化完成，开始构建UI...")
        except Exception as e:
            error(f"初始化ManualQuerySearch时出错: {str(e)}")
            traceback.print_exc()
            sys.exit(1)
    
    def load_config(self, config_path):
        """
        加载配置文件
        
        功能：
        - 读取YAML格式的配置文件
        - 解析为Python字典结构
        
        参数：
        - config_path: 配置文件路径
        
        返回值：
        - config: 包含配置信息的字典
        
        与用户界面联动：
        - 间接决定了WebUI的各项配置参数
        """
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config
    
    def retrieval_text(self, query):
        """
        文本检索模块接口【可定制】
        
        功能：
        - 使用BM25算法从文档集合中检索与查询相关的文档
        - BM25是基于词频和逆文档频率的经典检索算法
        
        参数：
        - query: 用户输入的查询字符串
        - context: 可选的上下文信息，默认为None
        
        返回值：
        - {answer: "str: 答案", document: "list: 相关文档"}

        与用户界面联动：
        - 当用户在Radio按钮中选择"bm25"并提交问题时，系统调用此方法
        - 检索结果将被process_query方法处理，最终显示在WebUI的相关文档区域
        
        配置相关：
        - 使用config.yaml中retrieval.bm25部分的参数配置（k1和b值）
        """
        if not query or not query.strip():
            return {
                "question": "",
                "document_id": []
            }
        
        # 使用初始化好的搜索器
        result = self.searcher.search(query)
        return result
    
    def retrieval_dpr(self, query):
        """
        dpr检索模块接口【可定制】
        
        功能：
        - 使用BERT模型进行语义检索，找到与查询在语义上相关的文档
        - 基于深度学习的向量检索方法，捕捉查询与文档的语义相似性
        
        参数：
        - query: 用户输入的查询字符串
        - context: 可选的上下文信息，默认为None
        
        返回值：
        - 检索到的相关文档列表
        
        与用户界面联动：
        - 当用户在Radio按钮中选择"dpr"并提交问题时，系统调用此方法
        - 检索结果将被process_query方法处理，最终显示在WebUI的相关文档区域
        
        配置相关：
        - 使用config.yaml中retrieval.dpr部分的模型配置
        """
        from dpr.query_process import query2doclist as q2d
        q2d = q2d(query, self.get_api_key(), "BAAI/bge-m3", "./cache/faiss/bgem3.faiss", 5)
        doc_id_list = q2d.query2doclist()
        return doc_id_list

    def retrieval_hybrid(self, query):
        """
        混合检索模块接口【可定制】
        
        功能：
        - 结合BM25和BERT的检索结果，利用多种检索方法的优势
        - 可以使用不同的融合方法（加权求和、倒数排名融合等）
        
        参数：
        - query: 用户输入的查询字符串
        - context: 可选的上下文信息，默认为None
        
        返回值：
        - 检索到的相关文档列表
        
        与用户界面联动：
        - 当用户在Radio按钮中选择"hybrid"并提交问题时，系统调用此方法
        - 检索结果将被process_query方法处理，最终显示在WebUI的相关文档区域
        
        配置相关：
        - 使用config.yaml中retrieval.hybrid部分的配置
        - weights参数决定BM25和BERT各自的权重
        - fusion_method决定融合方法（weighted_sum或reciprocal_rank等）
        """
        pass

    def get_api_key(self) -> str:
        """
        获取API密钥, silicon flower的API密钥
        """
        api_key = self.config.get("generation", {}).get("qwen", {}).get("api_key", "")
        if not api_key:
            raise ValueError("API密钥未配置")
        return api_key
    
    def retrieve_doc(self, idx_list: list[int]) -> list[str]:
        """根据索引值找到对应的文档"""
        try:
            doc_path = self.search_config.get("doc_path")
            log_message(f"正在读取文档文件: {doc_path}")

            # 转换为字符串列表进行匹配
            str_idx_list = [str(idx) for idx in idx_list]
            
            import pandas as pd
            df = pd.read_json(doc_path, lines=True, dtype={'doc_id': str})
            
            # 调试输出数据结构
            debug(f"数据文件包含列: {df.columns.tolist()}")
            debug(f"前3条数据示例:\n{df.head(3)}")
            
            # 检查必要字段是否存在
            if 'doc_id' not in df.columns or 'text' not in df.columns:
                raise KeyError("数据文件缺少doc_id或text字段")

            # 批量查询并保留顺序
            matched_df = df[df['doc_id'].isin(str_idx_list)]
            debug(f"匹配到{len(matched_df)}条记录，预期ID列表: {str_idx_list}")

            # 按输入顺序排序并去重
            ordered_docs = []
            seen = set()
            for idx in str_idx_list:
                if idx not in seen:
                    doc = matched_df[matched_df['doc_id'] == idx]
                    if not doc.empty:
                        ordered_docs.append(doc.iloc[0]['text'])
                        seen.add(idx)

            # 记录匹配结果
            found_ids = list(seen)
            if found_ids:
                log_message(f"成功匹配文档ID: {found_ids}")
            else:
                warning("未找到任何匹配文档")
            print(ordered_docs[0])
            return ordered_docs

        except Exception as e:
            error(f"文档检索失败: {str(e)}")
            traceback.print_exc()
            return []

    def rerank_results(self, query:str, document_id_list:list[int], top_n:int=1):
        """
        重排documnet_id_list的顺序,根据相关性
        input:
        - query: 用户输入的查询字符串
        - document_id_list: list[int], 文档索引列表,即document_ID
        - top_n: int, 需要返回的document_id数量
        return: top_n个文档的索引值list
        """
        api_token = self.get_api_key()
        document_content_list = self.retrieve_doc(document_id_list)
        query_document_list = [
            [query, document_content_list]
        ]
        reranker = ReRanker(api_token) 
        responses = reranker.async_send_requests_simple(
            query_document_list, use_progress_bar=False, concurrency=1
        )
        _, indices = reranker.extract_json(responses) # return: scores, indices
        return indices[0][:top_n] # 取前n个文档的索引值

    def generate_answer(self, query:str, context:str, n=5)->str:
        """
        使用QwenChatClient生成答案
        输入参数：
        - query: 用户输入的查询字符串
        - context: 检索到的相关文档
        - n: 生成答案的数量
        return:
        answer: 生成的答案文本
        """
        api_token = self.get_api_key()
        client = QwenChatClient(api_token=api_token)
        query_context_list = [(query, context)]
        results = client.batch_request_async_simple(
            query_context_list=query_context_list, 
            concurrency=1, model="Qwen/Qwen2.5-7B-Instruct", 
            n = n)
        extracted_answers_list = client.extract_answer(results)[0]
        # return the most frequent answer
        answer = max(set(extracted_answers_list), key=extracted_answers_list.count)
        return answer

    def format_document_display(self, document_ids):
        """格式化文档ID列表为显示文本"""
        if not document_ids:
            return "未找到相关文档"
            
        # 确保文档ID都是字符串类型
        str_doc_ids = [str(doc_id) for doc_id in document_ids]
            
        # 显示最多前5个文档ID
        doc_list_text = "，".join(str_doc_ids[:5])
            
        return f"related documents: {doc_list_text}"

    def process_query(self, query, retrieval_method):
        """
        处理用户查询的核心方法
        
        功能：
        - 作为WebUI与后端处理逻辑的桥梁
        - 协调各个模块（检索、重排序、生成）的工作流程
        
        参数：
        - query: 用户输入的查询字符串
        - retrieval_method: 用户选择的检索方法（"bm25"、"bert_base"或"hybrid"）
        
        返回值：
        - answer: 生成的答案文本
        - context_display: 格式化后的相关文档文本
        
        与用户界面联动：
        - 被submit_button.click事件直接调用
        - 接收用户输入并返回结果到界面显示
        - 是用户交互行为与系统处理逻辑之间的核心连接点
        
        工作流程：
        1. 根据用户选择的检索方法调用相应的检索接口
        2. 对检索结果进行重排序
        3. 基于重排序后的文档生成答案
        4. 格式化结果用于界面显示
        """
        # 输入检查
        if not query or not query.strip():
            return "请输入查询内容", "无相关文档"
            
        # 去除首尾空白字符
        query = query.strip()
        
        # 记录查询信息
        log_message(f"用户提交查询: '{query}', 使用检索方法: {retrieval_method}")
        
        try:
            # 根据选择的检索方法检索文档
            if retrieval_method == "bm25":
                retrieval_results = self.retrieval_text(query)
            elif retrieval_method == "dpr":
                retrieval_results = self.retrieval_dpr(query)
            else:  # hybrid
                retrieval_results = self.retrieval_hybrid(query)
            
            # 获取文档ID列表
            doc_ids = retrieval_results.get("document_id", [])
            
            # 如果没有找到相关文档，直接生成答案
            if not doc_ids:
                log_message(f"未找到与查询 '{query}' 相关的文档")
                answer = "Cannot find related documents"
                return "None", "Null"
            
            # 重排序
            reranked_doc_ids = doc_ids
            
            try:
                # 尝试检索文档内容
                retrieved_docs = self.retrieve_doc(doc_ids)
                if retrieved_docs:
                    answer = self.generate_answer(query, retrieved_docs[0])
                else:
                    # 如果retrieve_doc返回空列表，也使用备用回答
                    log_message(f"虽然找到了文档ID，但无法获取相关文档内容，列表为空")
                    raise ValueError("Cannot find related content of the query")
                    sys.exit(1)
            except Exception as e:
                # 捕获retrieve_doc可能抛出的异常
                log_message(f"检索文档时出错: {str(e)}")
                answer = self.generate_answer(query, f"query: {query}")
            
            # 格式化文档ID为显示文本
            context_display = self.format_document_display(reranked_doc_ids)
            
            return answer, context_display
            
        except Exception as e:
            # 捕获整个处理过程中的异常
            error_msg = f"处理查询时出现错误: {str(e)}"
            log_message(error_msg)
            import traceback
            traceback.print_exc()
            
            # 尝试直接生成答案，不依赖检索结果
            try:
                answer = self.generate_answer(query, "处理查询时出现错误，将直接回答问题。")
            except:
                answer = "很抱歉，处理您的查询时出现了错误，请稍后再试。"
                
            return answer, "处理查询时出现错误"

    def build_ui(self):
        """
        构建Gradio界面
        
        功能：
        - 创建和配置WebUI的各个组件
        - 设置组件之间的布局和交互逻辑
        
        返回值：
        - demo: 配置好的Gradio Blocks界面对象
        
        与用户界面联动：
        - 直接定义了用户能看到的所有界面元素
        - 设置了用户交互（如按钮点击）触发的回调函数
        
        界面组件：
        1. 标题：显示"智能问答系统"
        2. 问题输入框：用户输入查询的文本框
        3. 检索方法选择：单选按钮组，用户选择使用的检索方法
        4. 提交按钮：触发查询处理
        5. 回答显示区：展示系统生成的答案
        6. 相关文档显示区：展示检索到的相关文档
        
        交互流程：
        1. 用户在输入框中输入问题
        2. 用户选择检索方法（默认使用配置文件中指定的方法）
        3. 用户点击提交按钮
        4. 系统调用process_query处理查询
        5. 处理结果显示在回答区和相关文档区
        """
        try:
            debug("开始构建Gradio界面...")
            with gr.Blocks(title="问答系统") as demo:
                gr.Markdown("# 智能问答系统")
                
                with gr.Row():
                    with gr.Column(scale=3):
                        # 问题输入框 - 使用placeholder而非value，避免自动触发查询
                        query_input = gr.Textbox(
                            label="请输入您的问题", 
                            placeholder="例如：什么是机器学习？", 
                            lines=2,
                            value=""  # 确保初始值为空
                        )
                        
                        # 检索方法选择
                        retrieval_method = gr.Radio(
                            choices=["bm25", "dpr", "hybrid"],
                            value=self.retrieval_type,
                            label="检索方法",
                            info="选择使用的检索方法"
                        )
                        
                        # 提交按钮
                        submit_button = gr.Button("提交问题", variant="primary")
                        
                        # 添加示例问题按钮，而不是自动执行的示例
                        with gr.Accordion("示例问题", open=False):
                            sample1_btn = gr.Button("什么是机器学习？")
                            sample2_btn = gr.Button("深度学习和传统机器学习有什么区别？")
                    
                    with gr.Column(scale=4):
                        # 答案显示区域
                        answer_output = gr.Textbox(
                            label="answer", 
                            lines=6,
                            value="model loaded, please input question and click submit."
                        )
                        
                        # 相关文档显示区域
                        context_output = gr.Textbox(
                            label="related documents", 
                            lines=10,
                            value="waiting for input..."
                        )
                
                # 设置提交按钮功能
                submit_button.click(
                    fn=self.process_query,
                    inputs=[query_input, retrieval_method],
                    outputs=[answer_output, context_output]
                )
                
                # 示例问题按钮点击事件
                def set_example_1():
                    return "什么是机器学习？"
                
                def set_example_2():
                    return "深度学习和传统机器学习有什么区别？"
                
                sample1_btn.click(
                    fn=set_example_1,
                    inputs=[],
                    outputs=[query_input]
                )
                
                sample2_btn.click(
                    fn=set_example_2,
                    inputs=[],
                    outputs=[query_input]
                )
                
            debug("Gradio界面构建完成")
            return demo
        except Exception as e:
            error(f"构建UI时出错: {str(e)}")
            traceback.print_exc()
            raise

    def launch(self):
        """
        启动WebUI服务
        
        功能：
        - 根据配置决定是否启动WebUI
        - 启动Gradio服务器
        
        返回值：
        - 布尔值，表示是否成功启动WebUI
        
        与用户界面联动：
        - 控制WebUI是否向用户展示
        - 设置服务器端口，用户通过浏览器访问此端口查看界面
        
        配置相关：
        - 根据config.yaml中webui.presentation决定是否启动界面
        - 使用webui.port指定的端口号
        
        启动流程：
        1. 检查presentation配置是否为true
        2. 如为true，构建UI并启动服务器
        3. 如为false，输出提示信息
        
        用户使用：
        - 当用户运行main.py时，launch方法被调用
        - 如果启动成功，用户可在浏览器中访问http://localhost:端口号 查看界面
        """
        if self.presentation:
            try:
                log_message(f"开始构建WebUI界面...")
                demo = self.build_ui()
                log_message(f"WebUI界面构建完成，尝试在端口{self.port}上启动服务...")
                
                # 使用更多参数来确保稳定启动，使用多线程方式启动Gradio
                import threading
                def start_server():
                    try:
                        demo.launch(
                            server_port=self.port,
                            share=False,
                            inbrowser=True,  # 自动打开浏览器
                            show_error=True,  # 显示详细错误
                            server_name="0.0.0.0",  # 绑定到所有接口
                            prevent_thread_lock=True,  # 防止线程锁定
                            quiet=False  # 不要静默模式
                        )
                    except Exception as e:
                        error(f"Gradio服务器启动失败: {str(e)}")
                
                # 启动服务器线程
                server_thread = threading.Thread(target=start_server)
                server_thread.daemon = True  # 设置为守护线程
                server_thread.start()
                
                log_message(f"WebUI服务已启动，可以通过 http://localhost:{self.port} 访问")
                
                # 等待用户中断
                try:
                    while True:
                        import time
                        time.sleep(1)
                except KeyboardInterrupt:
                    log_message("用户中断，关闭WebUI服务...")
                
                return True
            except Exception as e:
                error(f"启动WebUI服务时出错: {str(e)}")
                traceback.print_exc()
                return False
        else:
            print("WebUI未启用。在config.yaml中将webui.presentation设置为true可启用WebUI。")
            return False


def create_webui(config_path):
    """
    创建并启动WebUI的便捷函数
    
    功能：
    - 创建WebUI实例
    - 启动界面服务
    
    参数：
    - config_path: 配置文件路径
    
    返回值：
    - 表示WebUI是否成功启动的布尔值
    
    与用户界面联动：
    - 是main.py调用的入口点
    - 在用户执行命令行指令时间接触发
    
    使用场景：
    - 用户运行 python main.py --config config.yaml 时
    - main.py解析参数后调用此函数
    - 函数创建WebUI实例并尝试启动界面
    """
    webui = WebUI(config_path)
    return webui.launch()

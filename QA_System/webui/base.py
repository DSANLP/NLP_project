import gradio as gr
import yaml
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
        self.config = self.load_config(config_path)
        self.port = self.config.get("webui", {}).get("port", 8080)
        self.presentation = self.config.get("webui", {}).get("presentation", True)
        self.retrieval_type = self.config.get("modules", {}).get("active", {}).get("retrieval", "hybrid")
    
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
    
    def retrieval_bm25(self, query, context=None):
        """
        BM25检索模块接口【可定制】
        
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
        pass
        return {"answer": "答案", "document_id": [1, 2]}
    
    def retrieval_bert_base(self, query, context=None):
        """
        bert_base检索模块接口【可定制】
        
        功能：
        - 使用BERT模型进行语义检索，找到与查询在语义上相关的文档
        - 基于深度学习的向量检索方法，捕捉查询与文档的语义相似性
        
        参数：
        - query: 用户输入的查询字符串
        - context: 可选的上下文信息，默认为None
        
        返回值：
        - 检索到的相关文档列表
        
        与用户界面联动：
        - 当用户在Radio按钮中选择"bert_base"并提交问题时，系统调用此方法
        - 检索结果将被process_query方法处理，最终显示在WebUI的相关文档区域
        
        配置相关：
        - 使用config.yaml中retrieval.bert_base部分的模型配置
        """
        pass
        return {"answer": "答案", "document_id": [1, 2]}

    def retrieval_hybrid(self, query, context=None):
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
        return {"answer": "答案", "document_id": [1, 2]}

    def get_api_key(self) -> str:
        """
        获取API密钥, silicon flower的API密钥
        """
        raise NotImplementedError("请在config.yaml中配置API密钥")
    
    def retrieve_doc(self, idx_list: list[int]) -> list[str]:
        """
        根据索引值找到对应的文档
        input: idx_list: list[int], 文档索引列表,即document_ID
        return: list[str], 对应的文档内容list
        """
        raise NotImplementedError("请完成=根据索引值找到对应的文档=retrieve_doc方法")
        
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
        results = client.batch_request_sync_simple(
            query_context_list=query_context_list, 
            concurrency=1, model="Qwen/Qwen2.5-7B-Instruct", 
            n = n)
        extracted_answers_list = client.extract_answer(results)[0]
        # return the most frequent answer
        answer = max(set(extracted_answers_list), key=extracted_answers_list.count)
        return answer

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
        # 根据选择的检索方法检索文档
        if retrieval_method == "bm25":
            retrieval_results = self.retrieval_bm25(query)
        elif retrieval_method == "bert_base":
            retrieval_results = self.retrieval_bert_base(query)
        else:  # hybrid
            retrieval_results = self.retrieval_hybrid(query)
        
        # 重排序
        reranked_results = self.rerank_results(query, retrieval_results)
        
        # 生成答案
        answer = self.generate_answer(query, reranked_results)
        
        # 展示检索结果和生成的答案
        context_display = "\n\n".join([f"【相关文档 {i+1}】\n{doc}" for i, doc in enumerate(reranked_results)])
        
        return answer, context_display

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
        with gr.Blocks(title="问答系统") as demo:
            gr.Markdown("# 智能问答系统")
            
            with gr.Row():
                with gr.Column(scale=3):
                    # 问题输入框
                    query_input = gr.Textbox(label="请输入您的问题", placeholder="例如：什么是机器学习？", lines=2)
                    
                    # 检索方法选择
                    retrieval_method = gr.Radio(
                        choices=["bm25", "bert_base", "hybrid"],
                        value=self.retrieval_type,
                        label="检索方法",
                        info="选择使用的检索方法"
                    )
                    
                    # 提交按钮
                    submit_button = gr.Button("提交问题", variant="primary")
                
                with gr.Column(scale=4):
                    # 答案显示区域
                    answer_output = gr.Textbox(label="回答", lines=6)
                    
                    # 相关文档显示区域
                    context_output = gr.Textbox(label="相关文档", lines=10)
            
            # 设置提交按钮功能
            submit_button.click(
                fn=self.process_query,
                inputs=[query_input, retrieval_method],
                outputs=[answer_output, context_output] # answer_output: 对应 answer, context_output: 对应 context_display
            )
        
        return demo
    
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
            demo = self.build_ui()
            demo.launch(server_port=self.port)
            return True
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

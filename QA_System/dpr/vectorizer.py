import numpy as np
import requests
import time
import asyncio
import aiohttp
from typing import List, Union, Dict, Tuple
import json
from tqdm import tqdm
import os

class doc_vectorizer:
    '''
    input:
    doc_id: document ids List[str]
    context: document texts List[str]
    api_key: silicone api key
    model_name: silicone model name

    output:
    doc_id: document ids List[str]
    vector: document vectors List[List[float32]]
    '''
    def __init__(self, doc_ids: List[str], contexts: List[str], api_key: str, model_name: str):
        self.doc_ids = doc_ids
        self.contexts = contexts
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = "https://api.siliconflow.cn/v1/embeddings"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.batch_size = 32
        self.max_retries = 3
        self.retry_delay = 1  # 秒

    def _make_request(self, texts: Union[str, List[str]], batch_size: int = 32) -> List[List[float]]:
        """
        向SILICONFLOW API发送同步请求并获取嵌入向量
        
        参数:
            texts: 单个文本字符串或字符串列表
            batch_size: 批处理大小
            
        返回:
            嵌入向量列表
        """
        if isinstance(texts, str):
            texts = [texts]
            
        all_embeddings = []
        
        # 批量处理请求
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            payload = {
                "model": self.model_name,
                "input": batch_texts,
                "encoding_format": "float"
            }
            
            retry_count = 0
            while retry_count < self.max_retries:
                try:
                    print(f"发送API请求，批次大小: {len(batch_texts)}，第{retry_count+1}次尝试...")
                    # 添加请求超时设置
                    response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=60)
                    response.raise_for_status()
                    data = response.json()
                    
                    # 提取嵌入向量
                    if len(batch_texts) == 1:
                        embeddings = [data['data'][0]['embedding']]
                    else:
                        # 根据索引排序，确保顺序一致
                        sorted_data = sorted(data['data'], key=lambda x: x['index'])
                        embeddings = [item['embedding'] for item in sorted_data]
                    
                    all_embeddings.extend(embeddings)
                    print(f"API请求成功，获取到 {len(embeddings)} 个向量，每个维度 {len(embeddings[0])}")
                    break
                    
                except requests.RequestException as e:
                    retry_count += 1
                    if retry_count >= self.max_retries:
                        print(f"批次 {i} 到 {i+batch_size} API请求失败，已重试{retry_count}次: {e}")
                        if hasattr(e, 'response') and e.response:
                            print(f"响应内容: {e.response.text}")
                        raise
                    else:
                        wait_time = self.retry_delay * (2 ** (retry_count - 1))  # 指数退避
                        print(f"批次 {i} 到 {i+batch_size} 请求失败，等待 {wait_time} 秒后重试({retry_count}/{self.max_retries})...")
                        print(f"错误信息: {str(e)}")
                        time.sleep(wait_time)
                
        return all_embeddings

    async def _async_make_request(self, session, batch_texts, batch_idx):
        """异步发送请求并获取嵌入向量"""
        payload = {
            "model": self.model_name,
            "input": batch_texts,
            "encoding_format": "float"
        }
        
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                async with session.post(self.api_url, headers=self.headers, json=payload) as response:
                    if response.status != 200:
                        text = await response.text()
                        raise Exception(f"API返回错误码: {response.status}, 响应内容: {text}")
                    
                    data = await response.json()
                    
                    # 提取嵌入向量
                    if len(batch_texts) == 1:
                        embeddings = [data['data'][0]['embedding']]
                    else:
                        # 根据索引排序，确保顺序一致
                        sorted_data = sorted(data['data'], key=lambda x: x['index'])
                        embeddings = [item['embedding'] for item in sorted_data]
                    
                    return batch_idx, embeddings
            
            except Exception as e:
                retry_count += 1
                if retry_count >= self.max_retries:
                    print(f"批次 {batch_idx} API请求失败，已重试{retry_count}次: {e}")
                    raise
                else:
                    print(f"批次 {batch_idx} 请求失败，正在重试({retry_count}/{self.max_retries})...")
                    await asyncio.sleep(self.retry_delay)

    async def _process_batches_async(self, max_concurrent_requests=5):
        """并发处理多个批次的文档"""
        if not self.contexts:
            return []
        
        results = [None] * ((len(self.contexts) + self.batch_size - 1) // self.batch_size)
        
        async with aiohttp.ClientSession() as session:
            # 创建任务队列
            tasks = []
            
            batch_count = (len(self.contexts) + self.batch_size - 1) // self.batch_size
            with tqdm(total=batch_count, desc="向量化进度") as pbar:
                for i in range(0, len(self.contexts), self.batch_size):
                    batch_texts = self.contexts[i:i+self.batch_size]
                    batch_idx = i // self.batch_size
                    
                    # 控制并发数量
                    while len(tasks) >= max_concurrent_requests:
                        # 等待一个任务完成
                        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                        # 更新任务列表
                        tasks = list(pending)
                        for task in done:
                            try:
                                idx, embeddings = await task
                                results[idx] = embeddings
                                pbar.update(1)
                            except Exception as e:
                                print(f"处理批次时出错: {e}")
                    
                    # 添加新任务
                    task = asyncio.create_task(self._async_make_request(session, batch_texts, batch_idx))
                    tasks.append(task)
                
                # 等待所有剩余任务完成
                while tasks:
                    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    # 更新任务列表
                    tasks = list(pending)
                    for task in done:
                        try:
                            idx, embeddings = await task
                            results[idx] = embeddings
                            pbar.update(1)
                        except Exception as e:
                            print(f"处理批次时出错: {e}")
        
        # 展平结果
        all_embeddings = []
        for batch in results:
            if batch:
                all_embeddings.extend(batch)
        
        return all_embeddings

    def vectorize(self, use_async=True, max_concurrent_requests=5) -> Tuple[List[str], List[List[float]]]:
        """
        对文档进行向量化
        
        参数:
            use_async: 是否使用异步处理
            max_concurrent_requests: 最大并发请求数
            
        返回:
            tuple: (doc_ids, vectors)
        """
        print(f"开始处理 {len(self.contexts)} 个文档...")
        
        # 获取嵌入向量
        if use_async:
            # 使用异步处理
            vectors = asyncio.run(self._process_batches_async(max_concurrent_requests))
        else:
            # 使用同步处理
            vectors = self._make_request(self.contexts, self.batch_size)
        
        # 确保doc_ids和vectors的长度一致
        if len(self.doc_ids) != len(vectors):
            raise ValueError(f"文档ID数量({len(self.doc_ids)})和向量数量({len(vectors)})不匹配")
            
        # 对向量进行L2归一化
        print("正在进行向量归一化...")
        vectors = np.array(vectors)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        zero_mask = norms == 0
        norms[zero_mask] = 1.0  # 避免除以零
        vectors = vectors / norms
        
        print(f"向量化完成！得到 {len(vectors)} 个向量。")
        return self.doc_ids, vectors.tolist()

    def process_large_dataset(self, chunk_size=1000, save_path=None):
        """
        分块处理大型数据集，并可以保存中间结果
        
        参数:
            chunk_size: 每个块的大小
            save_path: 保存中间结果的路径格式，如 'vectors_{}.json'
            
        返回:
            tuple: (doc_ids, vectors)
        """
        all_doc_ids = []
        all_vectors = []
        
        # 计算块数
        total_chunks = (len(self.doc_ids) + chunk_size - 1) // chunk_size
        
        for i in range(total_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(self.doc_ids))
            
            print(f"处理块 {i+1}/{total_chunks} (文档 {start_idx} 到 {end_idx})...")
            
            # 获取当前块的数据
            chunk_doc_ids = self.doc_ids[start_idx:end_idx]
            chunk_contexts = self.contexts[start_idx:end_idx]
            
            # 创建当前块的向量化器
            chunk_vectorizer = doc_vectorizer(
                chunk_doc_ids, 
                chunk_contexts, 
                self.api_key, 
                self.model_name
            )
            
            # 向量化当前块
            chunk_doc_ids, chunk_vectors = chunk_vectorizer.vectorize()
            
            # 添加到结果中
            all_doc_ids.extend(chunk_doc_ids)
            all_vectors.extend(chunk_vectors)
            
            # 保存中间结果
            if save_path:
                chunk_result = {
                    "doc_ids": chunk_doc_ids,
                    "vectors": chunk_vectors
                }
                with open(save_path.format(i), 'w', encoding='utf-8') as f:
                    json.dump(chunk_result, f)
                print(f"块 {i+1} 的结果已保存到 {save_path.format(i)}")
        
        return all_doc_ids, all_vectors

    def estimate_tokens(self, text):
        """
        估算文本的token数量（粗略估计，每个单词约1.3个token）
        """
        # 粗略估计：英文平均每个单词约1.3个token，中文每个字约1-2个token
        words = len(text.split())
        chars = len(text) - words  # 非空格字符，粗略计算中文字符
        return int(words * 1.3 + chars * 1.5)  # 粗略估计
    
    def split_long_text_with_overlap(self, text, max_tokens=7600, overlap_ratio=0.2):
        """
        将长文本分割成多个短文本，并添加重叠部分，每个短文本的token数不超过max_tokens
        
        参数:
            text: 需要分割的文本
            max_tokens: 每个分段的最大token数
            overlap_ratio: 重叠部分占比，默认20%
            
        返回:
            分割后的文本列表
        """
        # 如果文本估计token数小于限制，直接返回
        est_tokens = self.estimate_tokens(text)
        if est_tokens <= max_tokens:
            return [text]
            
        print(f"文本长度估计为 {est_tokens} tokens，需要分割(带重叠)")
        
        # 计算重叠token数
        overlap_tokens = int(max_tokens * overlap_ratio)
        effective_tokens = max_tokens - overlap_tokens
        
        # 按句子分割
        sentences = text.replace('\n', '. ').split('. ')
        chunks = []
        current_sentences = []
        current_tokens = 0
        last_overlap_sentences = []
        last_overlap_tokens = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_tokens = self.estimate_tokens(sentence)
            
            # 如果单个句子超过有效token限制，需要进一步分割
            if sentence_tokens > effective_tokens:
                # 如果当前chunk不为空，先保存
                if current_sentences:
                    chunks.append('. '.join(current_sentences))
                    # 记录重叠部分
                    overlap_count = 0
                    last_overlap_sentences = []
                    last_overlap_tokens = 0
                    for s in reversed(current_sentences):
                        s_tokens = self.estimate_tokens(s)
                        if last_overlap_tokens + s_tokens <= overlap_tokens:
                            last_overlap_sentences.insert(0, s)
                            last_overlap_tokens += s_tokens
                        else:
                            break
                    current_sentences = last_overlap_sentences.copy()
                    current_tokens = last_overlap_tokens
                
                # 按词分割超长句子
                words = sentence.split()
                sub_chunk = []
                sub_tokens = 0
                
                for word in words:
                    word_tokens = self.estimate_tokens(word)
                    if sub_tokens + word_tokens > effective_tokens:
                        if sub_chunk:
                            # 添加上一个chunk的重叠部分
                            if last_overlap_sentences and chunks:
                                full_chunk = '. '.join(last_overlap_sentences) + ' ' + ' '.join(sub_chunk)
                            else:
                                full_chunk = ' '.join(sub_chunk)
                            chunks.append(full_chunk)
                            
                            # 保存重叠部分
                            overlap_words = []
                            overlap_word_tokens = 0
                            for w in reversed(sub_chunk):
                                w_tokens = self.estimate_tokens(w)
                                if overlap_word_tokens + w_tokens <= overlap_tokens:
                                    overlap_words.insert(0, w)
                                    overlap_word_tokens += w_tokens
                                else:
                                    break
                            
                            sub_chunk = overlap_words.copy()
                            sub_tokens = overlap_word_tokens
                            last_overlap_sentences = []
                            last_overlap_tokens = 0
                    
                    sub_chunk.append(word)
                    sub_tokens += word_tokens
                
                if sub_chunk:
                    if last_overlap_sentences and chunks:
                        full_chunk = '. '.join(last_overlap_sentences) + ' ' + ' '.join(sub_chunk)
                    else:
                        full_chunk = ' '.join(sub_chunk)
                    chunks.append(full_chunk)
                    
                    # 更新当前chunk为重叠部分
                    current_sentences = []
                    current_tokens = 0
                    last_overlap_sentences = []
                    last_overlap_tokens = 0
                    for w in reversed(sub_chunk):
                        w_tokens = self.estimate_tokens(w)
                        if last_overlap_tokens + w_tokens <= overlap_tokens:
                            last_overlap_sentences.insert(0, w)
                            last_overlap_tokens += w_tokens
                        else:
                            break
                    current_sentences = last_overlap_sentences.copy()
                    current_tokens = last_overlap_tokens
            
            # 如果加上这个句子会超出有效限制，保存当前chunk，开始新chunk
            elif current_tokens + sentence_tokens > effective_tokens:
                if current_sentences:
                    chunks.append('. '.join(current_sentences))
                    
                    # 保存重叠部分
                    last_overlap_sentences = []
                    last_overlap_tokens = 0
                    for s in reversed(current_sentences):
                        s_tokens = self.estimate_tokens(s)
                        if last_overlap_tokens + s_tokens <= overlap_tokens:
                            last_overlap_sentences.insert(0, s)
                            last_overlap_tokens += s_tokens
                        else:
                            break
                            
                    current_sentences = last_overlap_sentences.copy()
                    current_tokens = last_overlap_tokens
                
                current_sentences.append(sentence)
                current_tokens += sentence_tokens
            
            # 否则加入当前chunk
            else:
                current_sentences.append(sentence)
                current_tokens += sentence_tokens
        
        # 保存最后一个chunk
        if current_sentences:
            chunks.append('. '.join(current_sentences))
            
        print(f"文本被分割为 {len(chunks)} 个带重叠的片段")
        return chunks
    
    def process_document_with_overlap(self, doc_id, text, overlap_ratio=0.2, use_weighted_avg=True):
        """
        处理单个文档，支持自动分割超长文档并添加重叠部分
        
        参数:
            doc_id: 文档ID
            text: 文档文本
            overlap_ratio: 重叠部分占比，默认20%
            use_weighted_avg: 是否使用加权平均，考虑片段长度
            
        返回:
            (doc_id, vector): 文档ID和对应的向量
        """
        # 分割长文档（带重叠）
        text_chunks = self.split_long_text_with_overlap(text, max_tokens=7600, overlap_ratio=overlap_ratio)
        
        chunk_lengths = [len(chunk) for chunk in text_chunks]
        chunk_tokens = [self.estimate_tokens(chunk) for chunk in text_chunks]
        
        print(f"文档 {doc_id} 被分割成 {len(text_chunks)} 个带重叠的片段")
        if len(text_chunks) > 1:
            print(f"片段平均长度: {sum(chunk_lengths)/len(chunk_lengths):.1f} 字符, 平均token数: {sum(chunk_tokens)/len(chunk_tokens):.1f} tokens")
            print(f"最大片段: {max(chunk_tokens)} tokens, 最小片段: {min(chunk_tokens)} tokens")
            print(f"重叠率: {overlap_ratio*100:.1f}%, 使用{'加权' if use_weighted_avg else '简单'}平均")
        
        if len(text_chunks) == 1:
            # 文档足够短，直接处理
            try:
                print(f"开始处理文档 {doc_id}，长度: {len(text_chunks[0])} 字符，约 {chunk_tokens[0]} tokens")
                embeddings = self._make_request(text_chunks[0], batch_size=1)
                if embeddings:
                    return doc_id, embeddings[0]
                else:
                    print(f"警告：文档 {doc_id} 获取嵌入向量失败")
                    return None, None
            except Exception as e:
                print(f"处理文档 {doc_id} 时出错: {e}")
                return None, None
        else:
            # 文档太长，分段处理后进行加权平均
            all_embeddings = []
            weights = []
            
            for i, chunk in enumerate(text_chunks):
                try:
                    est_tokens = chunk_tokens[i]
                    print(f"处理文档 {doc_id} 的第 {i+1}/{len(text_chunks)} 个片段，长度: {len(chunk)} 字符，约 {est_tokens} tokens")
                    embeddings = self._make_request(chunk, batch_size=1)
                    if embeddings:
                        all_embeddings.append(embeddings[0])
                        # 使用文本长度或token数作为权重
                        weights.append(est_tokens if use_weighted_avg else 1.0)
                        print(f"文档 {doc_id} 的第 {i+1} 个片段处理成功")
                except Exception as e:
                    print(f"处理文档 {doc_id} 的第 {i+1} 个片段时出错: {e}")
                    
                # 添加请求间隔，避免API限流
                if i < len(text_chunks) - 1:
                    print(f"等待2秒后处理下一个片段...")
                    time.sleep(2)
            
            if not all_embeddings:
                print(f"警告：文档 {doc_id} 的所有片段获取嵌入向量均失败")
                return None, None
                
            # 计算加权平均嵌入向量
            print(f"合并文档 {doc_id} 的 {len(all_embeddings)} 个片段向量 (使用{'加权' if use_weighted_avg else '简单'}平均)")
            
            if use_weighted_avg and sum(weights) > 0:
                # 归一化权重
                norm_weights = np.array(weights) / sum(weights)
                # 加权平均
                embeddings_array = np.array(all_embeddings)
                weighted_avg = np.zeros_like(embeddings_array[0])
                for i, embedding in enumerate(embeddings_array):
                    weighted_avg += embedding * norm_weights[i]
                avg_embedding = weighted_avg.tolist()
            else:
                # 简单平均
                avg_embedding = np.mean(np.array(all_embeddings), axis=0).tolist()
                
            # 归一化最终向量
            avg_embedding = np.array(avg_embedding)
            norm = np.linalg.norm(avg_embedding)
            if norm > 0:
                avg_embedding = (avg_embedding / norm).tolist()
                
            return doc_id, avg_embedding
            
    def process_documents_enhanced(self, chunk_size=10, overlap_ratio=0.2, use_weighted_avg=True):
        """
        增强版文档处理方法，支持自动处理超长文档并添加重叠部分
        
        参数:
            chunk_size: 每次处理的文档数量
            overlap_ratio: 重叠部分占比，默认20%
            use_weighted_avg: 是否使用加权平均，考虑片段长度
            
        返回:
            tuple: (doc_ids, vectors)
        """
        all_doc_ids = []
        all_vectors = []
        
        # 计算总文档数
        total_docs = len(self.doc_ids)
        print(f"开始处理 {total_docs} 个文档 (重叠率: {overlap_ratio*100:.1f}%, 使用{'加权' if use_weighted_avg else '简单'}平均)...")
        
        for i in range(0, total_docs, chunk_size):
            end_idx = min(i + chunk_size, total_docs)
            print(f"处理文档 {i+1} 到 {end_idx} (共 {total_docs} 个)...")
            
            for j in range(i, end_idx):
                doc_id = self.doc_ids[j]
                text = self.contexts[j]
                
                print(f"处理文档 {j+1}/{total_docs}: {doc_id} (大约 {self.estimate_tokens(text)} 个token)")
                result_id, vector = self.process_document_with_overlap(doc_id, text, overlap_ratio, use_weighted_avg)
                
                if result_id is not None and vector is not None:
                    all_doc_ids.append(result_id)
                    all_vectors.append(vector)
            
            print(f"已完成 {end_idx}/{total_docs} 个文档的处理")
        
        # 归一化向量
        if all_vectors:
            print("正在进行向量归一化...")
            vectors = np.array(all_vectors)
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            zero_mask = norms == 0
            norms[zero_mask] = 1.0  # 避免除以零
            vectors = vectors / norms
            all_vectors = vectors.tolist()
        
        print(f"处理完成！成功生成 {len(all_vectors)} 个文档向量")
        return all_doc_ids, all_vectors
    
    def estimate_processing_cost(self, sample_size=100):
        """
        估算处理所有文档的时间和成本
        
        参数:
            sample_size: 用于估算的样本大小
            
        返回:
            dict: 包含估算信息的字典
        """
        total_docs = len(self.doc_ids)
        
        # 限制样本大小，避免超出文档总数
        sample_size = min(sample_size, total_docs)
        
        # 计算样本中的总token数
        sample_tokens = 0
        for i in range(sample_size):
            text = self.contexts[i]
            sample_tokens += self.estimate_tokens(text)
        
        # 估算总token数
        estimated_total_tokens = (sample_tokens / sample_size) * total_docs
        
        # 估算API成本（按照标准BGE-M3计费规则：0.1元/1000tokens）
        estimated_cost_cny = (estimated_total_tokens / 1000) * 0.1
        
        # 估算处理时间（假设平均每秒处理10000 tokens，包括API延迟和计算开销）
        tokens_per_second = 10000  # 每秒处理的token数
        estimated_time_seconds = estimated_total_tokens / tokens_per_second
        
        # 格式化时间为小时:分钟:秒
        hours = int(estimated_time_seconds // 3600)
        minutes = int((estimated_time_seconds % 3600) // 60)
        seconds = int(estimated_time_seconds % 60)
        
        # 生成估算报告
        estimation = {
            "total_documents": total_docs,
            "sample_size": sample_size,
            "sample_tokens": sample_tokens,
            "avg_tokens_per_doc": sample_tokens / sample_size,
            "estimated_total_tokens": estimated_total_tokens,
            "estimated_cost_cny": estimated_cost_cny,
            "estimated_time": {
                "seconds": estimated_time_seconds,
                "formatted": f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            }
        }
        
        return estimation

class query_vectorizer:
    '''
    input: query str
    output: query_vector List[float32]
    '''
    def __init__(self, query: str, api_key: str, model_name: str):
        self.query = query
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = "https://api.siliconflow.cn/v1/embeddings"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.max_retries = 3
        self.retry_delay = 1  # 秒

    def _make_request(self, text: str) -> List[float]:
        """
        向SILICONFLOW API发送同步请求并获取查询文本的嵌入向量
        
        参数:
            text: 查询文本字符串
            
        返回:
            嵌入向量
        """
        payload = {
            "model": self.model_name,
            "input": text,
            "encoding_format": "float"
        }
        
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                print(f"发送查询API请求，第{retry_count+1}次尝试...")
                response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=60)
                response.raise_for_status()
                data = response.json()
                
                # 提取嵌入向量
                embedding = data['data'][0]['embedding']
                print(f"查询API请求成功，获取到向量，维度: {len(embedding)}")
                return embedding
                
            except requests.RequestException as e:
                retry_count += 1
                if retry_count >= self.max_retries:
                    print(f"查询API请求失败，已重试{retry_count}次: {e}")
                    if hasattr(e, 'response') and e.response:
                        print(f"响应内容: {e.response.text}")
                    raise
                else:
                    wait_time = self.retry_delay * (2 ** (retry_count - 1))  # 指数退避
                    print(f"查询请求失败，等待 {wait_time} 秒后重试({retry_count}/{self.max_retries})...")
                    print(f"错误信息: {str(e)}")
                    time.sleep(wait_time)

    def vectorize(self) -> List[float]:
        """
        对查询文本进行向量化
        
        返回:
            List[float]: 查询文本的向量表示
        """
        print(f"开始处理查询文本: {self.query[:100]}...")
        
        # 获取嵌入向量
        vector = self._make_request(self.query)
        
        # 对向量进行L2归一化
        print("正在进行向量归一化...")
        vector = np.array(vector)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        print("查询向量化完成！")
        return vector.tolist()
'''
if __name__ == "__main__":
    doc_ids = ["1", "2", "3"]
    contexts = ["Hello, world!", "This is a test.", "Another example."]
    api_key = "sk-nfizfypjawwnixaimezwbkxbhomzpuozlungqykzkwyporuk"
    model_name = "BAAI/bge-m3"

    vectorizer = doc_vectorizer(doc_ids, contexts, api_key, model_name)
    doc_ids, vectors = vectorizer.vectorize()
    print(doc_ids[0])
    print(vectors[0])
'''

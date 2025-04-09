import faiss_composer
import bgem3
import json
import os
import time
import numpy as np
import traceback
import argparse

def save_intermediate_results(doc_ids, vectors, save_path="./intermediate_results"):
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
    
    print(f"已保存中间结果到 {result_path}，包含 {len(doc_ids)} 个文档向量")
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
        
    return sorted(files, key=lambda x: int(x.split("_")[1].split(".")[0]), reverse=True)[0]

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="BGE-M3文档向量化工具")
    parser.add_argument("--overlap", type=float, default=0.2, help="文本分割的重叠率 (0-0.5 之间)")
    parser.add_argument("--chunk-size", type=int, default=10, help="每批处理的文档数量")
    parser.add_argument("--weighted", action="store_true", help="使用加权平均而不是简单平均")
    parser.add_argument("--load", action="store_true", help="是否自动加载最新的中间结果")
    return parser.parse_args()

def main():
    try:
        # 解析命令行参数
        args = parse_arguments()
        
        # 确保overlap在合理范围内
        overlap_ratio = max(0.0, min(0.5, args.overlap))
        chunk_size = max(1, args.chunk_size)
        use_weighted_avg = args.weighted
        
        print(f"参数设置: 重叠率={overlap_ratio:.2f}, 批处理大小={chunk_size}, 使用加权平均={use_weighted_avg}")
        
        # 读取文档
        api_key = "sk-nfizfypjawwnixaimezwbkxbhomzpuozlungqykzkwyporuk"
        model_name = "BAAI/bge-m3"
        doc_path = "./bgem3/processed_plain.jsonl"
        intermediate_path = "./intermediate_results"
        
        # 尝试加载之前的中间结果
        latest_file = get_latest_intermediate_file(intermediate_path)
        all_doc_ids = []
        all_vectors = []
        start_index = 0
        
        if latest_file and (args.load or input(f"发现之前的中间结果: {latest_file}，是否加载? (y/n): ").strip().lower() == 'y'):
            print(f"加载中间结果: {latest_file}")
            all_doc_ids, all_vectors = load_intermediate_results(f"{intermediate_path}/{latest_file}")
            print(f"已加载 {len(all_doc_ids)} 个文档向量")
        
        print("开始加载文档数据...")
        doc_jsonler = bgem3.doc_jsonler(doc_path)
        doc_ids, contexts = doc_jsonler.get_json_data()
        print(f"成功加载 {len(doc_ids)} 个文档")
        
        # 计算总文档数和预估的总token数
        total_docs = len(doc_ids)
        
        # 创建一个向量化器实例用于估算token
        temp_vectorizer = bgem3.doc_vectorizer([], [], api_key, model_name)
        total_tokens = sum(temp_vectorizer.estimate_tokens(text) for text in contexts[:min(100, total_docs)])
        avg_tokens_per_doc = total_tokens / min(100, total_docs)
        estimated_total_tokens = avg_tokens_per_doc * total_docs
        
        print(f"开始处理 {total_docs} 个文档，预估总token数约 {estimated_total_tokens:.0f}，平均每文档 {avg_tokens_per_doc:.1f} tokens")
        
        save_frequency = max(1, min(50, total_docs // 10))  # 每处理10%的文档保存一次，最少1个，最多50个
        print(f"每处理 {save_frequency} 个文档保存一次中间结果")
        
        # 如果已加载中间结果，确定开始处理的索引位置
        if all_doc_ids:
            # 查找已处理的最后一个文档在原始文档列表中的位置
            processed_doc_ids_set = set(all_doc_ids)
            for i in range(len(doc_ids)):
                if doc_ids[i] not in processed_doc_ids_set:
                    start_index = i
                    break
            print(f"继续从第 {start_index+1} 个文档开始处理（跳过 {start_index} 个已处理的文档）")
        
        processed_count = len(all_doc_ids)
        start_time = time.time()
        
        for i in range(start_index, total_docs, chunk_size):
            try:
                end_idx = min(i + chunk_size, total_docs)
                print(f"处理文档 {i+1} 到 {end_idx} (共 {total_docs} 个，已完成 {((i+processed_count-start_index)/total_docs*100):.1f}%)...")
                
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
                
                print(f"已完成 {len(all_doc_ids)}/{total_docs} 个文档的处理 (速度: {docs_per_second:.2f} 文档/秒)")
                
                # 定期保存中间结果，而不是每个批次都保存
                if len(all_doc_ids) % save_frequency == 0 or end_idx == total_docs:
                    save_intermediate_results(all_doc_ids, all_vectors)
                
                # 动态调整等待时间，根据文档处理速度
                if end_idx < total_docs:
                    # 如果处理速度快，减少等待时间；如果处理速度慢，增加等待时间
                    if docs_per_second > 0.5:  # 每2秒以上处理一个文档
                        wait_time = 1
                    else:
                        wait_time = 3
                    print(f"等待 {wait_time} 秒后处理下一批次...")
                    time.sleep(wait_time)
                
            except Exception as e:
                # 保存已处理的结果
                if all_doc_ids:
                    save_path = save_intermediate_results(all_doc_ids, all_vectors)
                    print(f"处理过程中出错，已保存中间结果到 {save_path}")
                print(f"错误详情: {str(e)}")
                traceback.print_exc()
                break
        
        # 处理完成后，保存到faiss索引
        if all_doc_ids:
            save_intermediate_results(all_doc_ids, all_vectors)
            print("正在保存到faiss索引...")
            faiss_composer.FaissSaver(all_doc_ids, all_vectors, "bgem3", "./faiss/").save()
            print("处理完成！")
    
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()

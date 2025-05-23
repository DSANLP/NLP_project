import requests
import asyncio
import aiohttp
from tqdm.asyncio import tqdm
import os


class ReRanker:
    def __init__(self, api_token, url="https://api.siliconflow.cn/v1/rerank"):
        self.url = url
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        }

    def send_request(
        self,
        query,
        documents,
        return_documents=False,
        max_chunks_per_doc=6000,
        overlap_tokens=100,
        model="BAAI/bge-reranker-v2-m3",
        top_n=5,
        **kwargs,
    ):
        payload = {
            "query": query,
            "documents": documents,
            "return_documents": return_documents,
            "max_chunks_per_doc": max_chunks_per_doc,
            "overlap_tokens": overlap_tokens,
            "model": model,
            "top_n": top_n,
        }
        response = requests.post(self.url, json=payload, headers=self.headers)
        return response.json()
    
    async def async_send_request(
        self,
        session,
        query,
        documents,
        return_documents=False,
        max_chunks_per_doc=1024,
        overlap_tokens=100,
        model="BAAI/bge-reranker-v2-m3",
        top_n=5,
        **kwargs,
    ):
        payload = {
            "query": query,
            "documents": documents,
            "return_documents": return_documents,
            "max_chunks_per_doc": max_chunks_per_doc,
            "overlap_tokens": overlap_tokens,
            "model": model,
            "top_n": top_n,
        }
        async with session.post(self.url, json=payload, headers=self.headers) as response:
            if response.status != 200:
                raise Exception(f"Request failed with status code {response.text}")
            return await response.json()

    async def async_send_requests(self, query_document_list, use_progress_bar=False, concurrency=5, **kwargs):
        async with aiohttp.ClientSession() as session:
            sem = asyncio.Semaphore(concurrency)
            
            async def sem_task(query, documents):
                async with sem:
                    return await self.async_send_request(session, query, documents, **kwargs)
            
            tasks = [sem_task(query, documents) for query, documents in query_document_list]
            
            if use_progress_bar:
                responses = []
                for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing"):
                    responses.append(await coro)
            else:
                responses = await asyncio.gather(*tasks)
        return responses
    
    def async_send_requests_simple(self, query_document_list, use_progress_bar=False, concurrency=5, **kwargs):
        # This function is a simple wrapper around async_send_requests
        return asyncio.run(
            self.async_send_requests(
                query_document_list, use_progress_bar=use_progress_bar, concurrency=concurrency, **kwargs
            )
        )
        
    def extract_json(self, responses):
        """
        return a list of scores and indices
        indices: the rank of the given document
        return example:
        scores = [[0.9, 0.8, 0.7], [0.9, 0.8, 0.7], [0.9, 0.8, 0.7]]
        indices = [[0, 1, 2], [0, 1, 2], [0, 1, 2]]
        """
        scores = []
        indices = []
        for response in responses:
            s, i = [], []
            results = response.get("results", [])
            for result in results:
                s.append(result["relevance_score"])
                i.append(result["index"])
            scores.append(s)
            indices.append(i)
        return scores, indices

if __name__ == "__main__":
    # Example usage
    api_token = "<api token>"
    api_token = (
        open(
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "silicon_api.key"),
            "r",
        )
        .read()
        .strip()
    )
    
    
    # =============== Example: async and extract ===============
    query_document_list = [
        ["apple", ["kobe "*20000, "apple and juice"*20000, "fruit", "vegetable", "banana", "orange"]],
        ["Banana", ["kobe", "apple", "fruit", "vegetable", "banana", "orange"]],
        ["Orange", ["kobe", "apple", "fruit", "vegetable", "banana", "orange"]],
    ]
    # use_progress_bar=True cannot keep the same order
    # use_progress_bar=False can keep the same order
    reranker = ReRanker(api_token)
    # responses = asyncio.run(reranker.async_send_requests(
    #     query_document_list, use_progress_bar=False, concurrency=5))    
    responses = reranker.async_send_requests_simple(
        query_document_list, use_progress_bar=False, concurrency=5, top_n = 5
    )
    scores, indices = reranker.extract_json(responses)
    print(scores)
    print(indices)
    # =========================================================
    
    
    
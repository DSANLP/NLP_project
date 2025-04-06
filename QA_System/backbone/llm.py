import os
import requests
import asyncio
import aiohttp
from tqdm import tqdm
import re

class QwenChatClient:
    def __init__(self, api_token, base_url="https://api.siliconflow.cn/v1/chat/completions"):
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
        self.clear_proxies()
    
    def clear_proxies(self):
        os.environ.pop("HTTP_PROXY", None)
        os.environ.pop("HTTPS_PROXY", None)
        os.environ.pop("http_proxy", None)
        os.environ.pop("https_proxy", None)
        os.environ.pop("ALL_PROXY", None)
        os.environ.pop("all_proxy", None)

    def request(self, query, context, **kwargs):
        prompt_type = kwargs.get("prompt_type", "instruct")
        messages = self.prompt(query, context, type=prompt_type)
        payload = {
            "model": kwargs.get("model", "Qwen/Qwen2.5-7B-Instruct"),
            "stream": kwargs.get("stream", False),
            "max_tokens": kwargs.get("max_tokens", 4096),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.7),
            "top_k": kwargs.get("top_k", 50),
            "frequency_penalty": kwargs.get("frequency_penalty", 0.5),
            "n": kwargs.get("n", 1), 
            "response_format": kwargs.get("response_format", {"type": "text"}),
            "messages": messages,
            "tools": [],
        }

        response = requests.post(self.base_url, json=payload, headers=self.headers)
        if response.status_code != 200:
            raise Exception(f"Failed to request QwenChat API: {response.text}")
        return response.json()
    
    def batch_requests(self, query_context_list, **kwargs):
        results = []
        for query, context in query_context_list:
            prompt_type = kwargs.get("prompt_type", "instruct")
            messages = self.prompt(query, context, type=prompt_type)
            result = self.request(messages, **kwargs)
            results.append(result)
        return results

    async def async_request(self, query, context, **kwargs):
        messages = self.prompt(query, context, type=kwargs.get("prompt_type", "instruct"))
        payload = {
            "model": kwargs.get("model", "Qwen/Qwen2.5-7B-Instruct"),
            "stream": kwargs.get("stream", False),
            "max_tokens": kwargs.get("max_tokens", 4096),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.7),
            "top_k": kwargs.get("top_k", 50),
            "frequency_penalty": kwargs.get("frequency_penalty", 0.5),
            "n": kwargs.get("n", 1), # answer the same question n times
            "response_format": kwargs.get("response_format", {"type": "text"}),
            "messages": messages,
            "tools": [],
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(self.base_url, json=payload, headers=self.headers) as response:
                if response.status != 200:
                    text = await response.text()
                    raise Exception(f"Failed to request QwenChat API: {text}")
                return await response.json()

    async def async_batch_request(self, query_context_list, concurrency=5, **kwargs):
        semaphore = asyncio.Semaphore(concurrency)
        async def sem_request(query, context):
            async with semaphore:
                return await self.async_request(query, context, **kwargs)
        tasks = [asyncio.create_task(sem_request(query, context)) for query, context in query_context_list]
        return await asyncio.gather(*tasks)

    async def async_batch_request_with_progress(self, query_context_list, concurrency=5, **kwargs):
        semaphore = asyncio.Semaphore(concurrency)
        async def sem_request(query, context):
            async with semaphore:
                return await self.async_request(query, context, **kwargs)
        tasks = [asyncio.create_task(sem_request(query, context)) for query, context in query_context_list]
        results = []
        pbar = tqdm(total=len(tasks), desc="Processing")
        for task in asyncio.as_completed(tasks):
            result = await task
            results.append(result)
            pbar.update(1)
        pbar.close()
        return results
    
    def batch_request_async_simple(self, query_context_list=None, query_context_path=None, concurrency=5, **kwargs):
        """
        """
        if query_context_list is not None:
            return asyncio.run(self.async_batch_request(query_context_list, concurrency, **kwargs))
        if query_context_path is not None:
            query_context_list = self.load_data(query_context_path)
            return asyncio.run(self.async_batch_request(query_context_list, concurrency, **kwargs))
        raise ValueError("query_context_list or query_context_path should be provided.")

    
    def prompt(self, query, context, type="instruct"):
        """
        do prompt engineer
        input: messages, type
        return: system prompt, user prompt
        """
        if type == "instruct":
            system_prompt = (
                "You are a helpful and friendly assistant. "
                "Please provide a brief answer "
                "to the following question."
            )
            # answer the query based on context
            user_prompt = (
                'Instructions: Please provide a concise and reliable answer with several keywords only to the query based on the given context. '
                'Ensure your response is directly supported by the context. Enclose your final answer within {{ }}.\n\n'
                f'Context:\n{context}\n\n'
                f'Query:\n{query}'
            )
        elif type == "CoT":
            system_prompt = (
            "You are a helpful assistant trained to answer questions using a Chain-of-Thought approach. "
            "Provide step-by-step reasoning and conclude with a concise answer."
            )
            # answer the query based on context
            user_prompt = (
                'Instructions: Please provide a concise and reliable answer with several keywords only to the query based on the given context. '
                'Ensure your response is directly supported by the context. Enclose your final answer within {{ }}.\n\n'
                f'Context:\n{context}\n\n'
                f'Query:\n{query}'
            )
        messages = [
            {
                "content": system_prompt,
                "role": "system"
            },
            {
                "content": user_prompt,
                "role": "user"
            }
        ]
        return messages
    
    def extract_answer(self, responses):
        """
        extract answers from responses
        responses have attribute choices, each choice have index, content
        extracted_answers: [[answer1, answer2], [answer1, answer2]]  # n=2
        """
        extracted_answers = []
        for response in tqdm(responses):
            choices = response.get("choices", [])
            answers = []
            for choice in choices:
                message = choice.get("message", {})
                content = message.get("content", "")
                answer = re.search(r"\{\{(.+?)\}\}", content)
                answers.append(answer.group(1) if answer else "")
            extracted_answers.append(answers)
        return extracted_answers
    
    def load_data(self, file_path):
        """
        load the query_context_list from file_path
        """
        pass


if __name__ == "__main__":
    api_token = "sk-****" # get api_key from https://cloud.siliconflow.cn/models
    api_token = (
        open(
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "silicon_api.key"),
            "r",
        )
        .read()
        .strip()
    )

    query = "What is the best Chinese large model?"
    context = "The Chinese large model industry is growing rapidly and has attracted a lot of attention. The industry is expected to face both opportunities and challenges in the coming years. The best model is Qwen."

    # # synchronous request
    # # we do not recommend using synchronous request function
    # response = client.request(query, context, model="Qwen/Qwen2.5-7B-Instruct")
    # print(response)

    # # ==========Asynchronous example==========
    # async def main(query_context_list):
    #     # Batch async request with a max concurrency
    #     batch_results = await client.async_batch_request_with_progress(query_context_list, concurrency=5, model="Qwen/Qwen2.5-7B-Instruct", n = 3)
    #     print("Async batch results:", batch_results)
    #     print("extracted answers:", client.extract_answer(batch_results))

    # query_context_list = [(query, context) for _ in range(2)]
    # asyncio.run(main(query_context_list))
    # # ==========Asynchronous example==========

    # # test prompt method
    # print(client.prompt(query, context, type="normal"))

    # ==========Batch request without progress(simple)==========
    client = QwenChatClient(api_token=api_token)
    query_context_list = [("What is the best Chinese large model?", 
                           "The Chinese large model industry is growing rapidly and has attracted a lot of attention. The industry is expected to face both opportunities and challenges in the coming years. The best model is Qwen."),
                          ("What is the best Chinese large model?",
                            "The Chinese large model industry is growing rapidly and has attracted a lot of attention. The industry is expected to face both opportunities and challenges in the coming years. The best model is deepseek."),
                            ("What is the best Chinese large model?",
                             "The Chinese large model industry is growing rapidly and has attracted a lot of attention. The industry is expected to face both opportunities and challenges in the coming years. The best model is hunyuan."),
                          ]
    results = client.batch_request_async_simple(
        query_context_list=query_context_list, 
        concurrency=5, model="Qwen/Qwen2.5-7B-Instruct", 
        n = 3)
    extracted_answers = client.extract_answer(results)
    # print("batch results:", results)
    print("extracted answers:", extracted_answers)
    # ==========Batch request with progress(simple)==========

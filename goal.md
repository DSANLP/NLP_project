！！先分成以下几个步骤完成第一阶段的任务，`data preprocess`， `BM25`，`dense vector and FAISS`，`Qwen and prompt`，`UI`，`Hybrid`，大家选一下想做什么，然后去看相关方法，下次会议分享方法，代码也可以开始写了。后面还有选读论文，可以选好后，发群里。

**async def**  

## data preprocess

对`documents.jsonl `中的html格式内容预处理，转换成ai友好的md格式, 存在`.json`文件里, 别用`.csv`

`\n` `<\br>`

## compulsory

1. Keyword Matching-Based Retrieval: **BM25(推荐了解这个)** or TF-IDF
2. Vector Space Model-Based Retrieval: Word2Vec, GloVe, or FastText 

3. Answer Generation via Prompting Large Language Models (LLM): 
   * prompt engineering
   * give the answer in `\{answer}\`

restrict the usage to `Qwen/Qwen2.5-7B-Instruct ` as the LLM for generating answers via prompting. use API



1. UI: 
   * User Input for Questions, 
   * Retrieved Documents, 
   * Answer Display

Remark: 注意`train set`, `val set`, `test set`

1. Advanced: 
   * Based on dense method  utilizing Approximate Nearest Neighbor (ANN) search algorithms, such as FAISS

## optional

You should do experiments, explanation of why the method is effective, detailing your exploration process, comparing the performance of the new approach

*  Hybrid Retrieval techniques  这是最常用的方法

*  Hybrid: input : [query, one doc] -> score * 10

* Related Work (Reference Papers): **！！大家选择一两篇下面论文看一看，下次分享论文提出的方法，主要是学习方法，基本上不实现，不用太花时间！！**
* 1. Reading Wikipedia to Answer Open-Domain Questions （较难）
  2. End-to-End Open-Domain Question Answering with BERTserini （简单）
  3. REALM: Retrieval-Augmented Language Model Pre-Training （简单）
  4. Dense Passage Retrieval for Open-Domain Question Answering （较难）
  5. Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering （简单）
  6. Qwen2.5 Technical Report （没必要看）





* 目前最速的RAG方法<https://paperswithcode.com/paper/bm25s-orders-of-magnitude-faster-lexical>
* 排行榜<https://paperswithcode.com/task/retrieval>

  

## Hand in

* code
* Written Report 
* Flowchart diagram 
* Results analysis
* UI
* Demo Video
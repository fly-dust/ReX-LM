import bm25s
from bm25s.hf import BM25HF
import os 
import json
from openai import OpenAI


REX_MODE = True

# vLLM: python -m vllm.entrypoints.openai.api_server --port 8000 --model Qwen/Qwen2-1.5B-Instruct --dtype auto --api-key token-vllm2024
base_url = "http://localhost:8000/v1"
api_key = "token-vllm2024"
model_name = "Qwen/Qwen2-1.5B-Instruct"
# model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"

hf_data_name = "Magpie-Align/Llama-3-Magpie-Pro-1M-v0.1"
index_name = "Llama-3-Magpie-Pro-1M-v0.1" 
hf_repo_name = f"yuchenlin/BM25S_index_{index_name}"

# you can specify revision and load_corpus=True if needed
retriever = BM25HF.load_from_hub(
    hf_repo_name, revision="main", load_corpus=True
) 

# Query the corpus
query = "Who is Donald Trump?"


rex_chat_history = []
rex_chat_history.append({"role": "system", "content": "You are a helpful assistant. Please help me with the following related questions."})
if REX_MODE:
    # Tokenize the query
    query_tokens = bm25s.tokenize(query, show_progress=False)

    # Get top-k results as a tuple of (doc ids, scores). Both are arrays of shape (n_queries, k)
    results, scores = retriever.retrieve(query_tokens, k=3)

    for i in range(results.shape[1]):
        doc, score = results[0, i], scores[0, i]
        # print(f"Rank {i+1} (score: {score:.2f}): {doc}") 
        rex_query = doc["query"]
        rex_response = doc["response"]
        rex_chat_history.append({"role": "user", "content": rex_query})
        rex_chat_history.append({"role": "assistant", "content": rex_response})

rex_chat_history.append({"role": "user", "content": query}) 

client = OpenAI(api_key=api_key, base_url=base_url)
 # print(f"Requesting chat completion from OpenAI API with model {model}")
messages = rex_chat_history

print(json.dumps(messages, indent=2))

temperature = 0.5
max_tokens = 1024
top_p = 1.0
n = 1  
presence_penalty = 0.0
kwargs = {}
response = client.chat.completions.create(
    model=model_name, 
    messages=messages,
    temperature=temperature,
    max_tokens=max_tokens,
    top_p=top_p,
    n=n,  
    presence_penalty=presence_penalty,
    **kwargs,
)

# print(f"Received response from OpenAI API with model {model}")
contents = []
for choice in response.choices:
    # Check if the response is valid
    if choice.finish_reason not in ['stop', 'length', 'eos']:
        if 'content_filter' in choice.finish_reason:
            contents.append("Error: content filtered due to OpenAI policy. ")
        else:
            raise ValueError(f"OpenAI Finish Reason Error: {choice.finish_reason}")
    contents.append(choice.message.content.strip())

print(contents)
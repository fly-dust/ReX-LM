from datasets import load_dataset
from transformers import Qwen2Tokenizer
import bm25s
import sys
import json
import random
from tqdm import tqdm
from bm25s.hf import BM25HF

def retrieve_nearest_datapoints(query, retriever, k=5):    
    query_tokens = bm25s.tokenize(query, show_progress=False)
    results, scores = retriever.retrieve(query_tokens, k=k, show_progress=False)
    
    retrieved_data = []
    for i in range(results.shape[1]):
        doc, score = results[0, i], scores[0, i]
        rex_query = doc["query"]
        rex_response = doc["response"]
        retrieved_data.append({
            "query": rex_query,
            "response": rex_response,
            "score": score
        })
    
    return retrieved_data

# Load Retriever
hf_repo_name = f"yuchenlin/BM25S_index_Llama-3-Magpie-Pro-1M-v0.1"
retriever = BM25HF.load_from_hub(
    hf_repo_name, revision="main", load_corpus=True
)

# Load dataset
magpie_pro_300k = load_dataset("Magpie-Align/Magpie-Pro-300K-Filtered", split = "train")
# Get first 50k samples
magpie_pro_50k = magpie_pro_300k.select(range(50000))
save_dir = "extracted_data_50k.jsonl"

print("Retrieving nearest datapoints...")
count = 0
with open(save_dir, "w") as f:
    for idx, data in enumerate(tqdm(magpie_pro_50k)):
        uuid = data['uuid']
        query = data['conversations'][0]['value']
        response = data['conversations'][1]['value']
        retrieved_data = retrieve_nearest_datapoints(query, retriever=retriever, k=random.randint(1, 4))
        # Check if there is any retrieved data = query
        retrieved_conversations = []
        scores = []
        for i in range(len(retrieved_data)):
            if retrieved_data[i]['query'] == query:
                print("Retrieved data = query")
                count += 1
                continue
            else:
                retrieved_conversations.append({"from": "human", "value": retrieved_data[i]['query']})
                retrieved_conversations.append({"from": "gpt", "value": retrieved_data[i]['response']})
                scores.append(str(retrieved_data[i]['score']))
        
        sharegpt_conversation = retrieved_conversations + [{"from": "human", "value": query}, {"from": "gpt", "value": response}]
        save_entry = {
            "query": query,
            "response": response,
            "sharegpt_conversation": sharegpt_conversation,
            "scores": scores
        }
        if idx%20 == 0:
            print(f"Entry {idx} saved!")
        f.write(json.dumps(save_entry) + "\n")

print(f"Total number of retrieved data = query: {count}")
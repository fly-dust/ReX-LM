import os
import bm25s
from bm25s.hf import BM25HF
import Stemmer  # optional: for stemming
from datasets import load_dataset
from tqdm import tqdm 

# hf_data_name = "Magpie-Align/Llama-3-Magpie-Air-3M-v0.1"
# index_name = "Llama-3-Magpie-Air-3M-v0.1" 

hf_data_name = "Magpie-Align/Llama-3-Magpie-Pro-1M-v0.1"
index_name = "Llama-3-Magpie-Pro-1M-v0.1" 

# Create your corpus here
# corpus = [
#     "a cat is a feline and likes to purr",
#     "a dog is the human's best friend and loves to play",
#     "a bird is a beautiful animal that can fly",
#     "a fish is a creature that lives in water and swims",
# ]

data = load_dataset(hf_data_name, split="train")
 
corpus_records = []
corpus_tokens_to_index = []
for item in tqdm(data, desc="Loading data"):
    uuid = item["uuid"]
    query = item["conversations"][0]["value"]
    response = item["conversations"][1]["value"] 
    if item['min_similar_uuid'] is None or item['min_similar_uuid'] == uuid:
        # corpus.append(response)
        corpus_records.append({"id": uuid, "query": query, "response": response})
        corpus_tokens_to_index.append(query)


# optional: create a stemmer
stemmer = Stemmer.Stemmer("english")

# Tokenize the corpus and only keep the ids (faster and saves memory)
corpus_tokens = bm25s.tokenize(corpus_tokens_to_index, stopwords="en", stemmer=stemmer)

# Create the BM25 model and index the corpus
retriever = BM25HF(corpus=corpus_records)
retriever.index(corpus_tokens)
hf_token = os.environ["HF_TOKEN"]
retriever.save_to_hub(f"yuchenlin/BM25S_index_{index_name}", token=hf_token, corpus=corpus_records)


# # You can save the arrays to a directory...
# retriever.save(index_name)

# You can save the corpus along with the model
# retriever.save(index_name, corpus=corpus)


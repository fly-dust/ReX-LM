import bm25s
from bm25s.hf import BM25HF

hf_data_name = "Magpie-Align/Llama-3-Magpie-Pro-1M-v0.1"
index_name = "Llama-3-Magpie-Pro-1M-v0.1" 
hf_repo_name = f"yuchenlin/BM25S_index_{index_name}"

# you can specify revision and load_corpus=True if needed
retriever = BM25HF.load_from_hub(
    hf_repo_name, revision="main", load_corpus=True
) 

# Query the corpus
query = "Who is Donald Trump?"

# Tokenize the query
query_tokens = bm25s.tokenize(query)

# Get top-k results as a tuple of (doc ids, scores). Both are arrays of shape (n_queries, k)
results, scores = retriever.retrieve(query_tokens, k=3)

for i in range(results.shape[1]):
    doc, score = results[0, i], scores[0, i]
    print(f"Rank {i+1} (score: {score:.2f}): {doc}")
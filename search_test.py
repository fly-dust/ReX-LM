# ...and load them when you need them
import bm25s
import Stemmer  # optional: for stemming

hf_data_name = "Magpie-Align/Llama-3-Magpie-Air-3M-v0.1"
index_name = "Llama-3-Magpie-Air-3M-v0.1" # 

# hf_data_name = "Magpie-Align/Llama-3-Magpie-Pro-1M-v0.1"
# index_name = "Llama-3-Magpie-Pro-1M-v0.1" 

# hf_data_name = "Magpie-Align/Magpie-Pro-300K-Filtered"
# index_name = "Magpie-Pro-300K-Filtered" 


retriever = bm25s.BM25.load(index_name, load_corpus=True)
# set load_corpus=False if you don't need the corpus

# optional: create a stemmer
stemmer = Stemmer.Stemmer("english")

# Query the corpus
query = "what is the capital of australia"
query_tokens = bm25s.tokenize(query, stemmer=stemmer)

# Get top-k results as a tuple of (doc ids, scores). Both are arrays of shape (n_queries, k)
# results, scores = retriever.retrieve(query_tokens, corpus=corpus, k=2)
results, scores = retriever.retrieve(query_tokens, k=10)

for i in range(results.shape[1]):
    doc, score = results[0, i], scores[0, i]
    print(f"Rank {i+1} (score: {score:.2f}): {doc}")

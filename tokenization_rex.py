from transformers import Qwen2Tokenizer
import bm25s
from bm25s.hf import BM25HF
# import Stemmer  # optional: for stemming

# define the RexQwen2Tokenizer class that is a subclass of Qwen2Tokenizer

class RexQwen2Tokenizer(Qwen2Tokenizer): 
    # to simplify the __init__ method 
    def __init__(
        self,
        rex_index_name = "Llama-3-Magpie-Pro-1M-v0.1",
        rex_size = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.rex_index_name = rex_index_name 
        hf_repo_name = f"yuchenlin/BM25S_index_{self.rex_index_name}"
        self.retriever = BM25HF.load_from_hub(
            hf_repo_name, revision="main", load_corpus=True
        ) 
        self.rex_size = rex_size
        self.user_prefix = "<|im_start|>user"

    def _rex_query(self, query):
        k = self.rex_size
        query_tokens = bm25s.tokenize(query, show_progress=False)
        results, scores = self.retriever.retrieve(query_tokens, k=k)
        rex_chat_history = []
        for i in range(results.shape[1]):
            doc, score = results[0, i], scores[0, i]
            rex_query = doc["query"]
            rex_response = doc["response"]
            rex_chat_history.append({"role": "user", "content": rex_query})
            rex_chat_history.append({"role": "assistant", "content": rex_response})
        rex_chat_history_tokens = self.apply_chat_template(rex_chat_history, tokenize=False, add_generation_prompt=False)
        start_user = rex_chat_history_tokens.index(self.user_prefix)
        rex_chat_history_tokens = rex_chat_history_tokens[start_user:]
        return rex_chat_history_tokens



    # redefine the _tokenize method
    def tokenize(self, text, **kwargs):   
        # find the index for first user query 
        if self.user_prefix not in text or self.rex_size < 1:
            # the query is not wrapped with chat template yet 
            # raise NotImplementedError
            return super().tokenize(text, **kwargs)
        start_index = text.index(self.user_prefix)
        rex_chat_history_tokens = self._rex_query(text[start_index+len(self.user_prefix):])
        rex_text = text[:start_index] + rex_chat_history_tokens + text[start_index:]
        # print(rex_text)
        tokens = super().tokenize(rex_text, **kwargs)
        return tokens



if __name__ == "__main__":
    from transformers import AutoTokenizer
    model_path = "yuchenlin/Rex-v0.1-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, rex_size=3)
    messages = [
        {"role": "user", "content": "Who is Yuchen Lin?"},
    ]
    query = tokenizer.apply_chat_template(messages, tokenize=False) 
    print(tokenizer.tokenize(query))

    # detokenize
    text = tokenizer.decode(tokenizer.tokenize(query))
    print(text)
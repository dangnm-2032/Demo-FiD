from haystack import BaseComponent
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import numpy as np
from copy import deepcopy

class ReRanker(BaseComponent):
    outgoing_edges = 1

    def __init__(self, 
                 model_name_or_path: str = 'llmrails/ember-v1',
                 device: str = "cpu"):
        super().__init__()
        self.device = device
        self.model = SentenceTransformer(model_name_or_path)
        self.model.to(self.device)
    
    def run(self, query, documents):
        inputs = [query] + list(doc.content for doc in documents)

        embeddings = self.model.encode(inputs)
        query_embed = embeddings[0]
        docs_embed = embeddings[1:]
        scores = []
        for doc_embed in docs_embed:
            score = cos_sim(query_embed, doc_embed)
            scores.append(score.tolist()[0][0])
        documents = np.array(documents)
        scores = np.array(scores)

        idx_sorted = np.argsort(scores)
        documents = documents[idx_sorted].tolist()
        scores = scores[idx_sorted].tolist()
        
        output = {"top2_docs": documents[-2:],
                  "top2_docs_score": scores[-2:]}
  
        return output, "output_1"
    
    def run_batch(self, queries: List[str], my_optional_param: Optional[int]):
        # process the inputs
        output = {"my_output": ...}
        return output, "output_1"

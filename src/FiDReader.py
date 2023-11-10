from haystack import BaseComponent
from src.FiD_T5 import FiD
from transformers import AutoTokenizer
from typing import List, Optional
import re

class FiDReader(BaseComponent):
    outgoing_edges = 1

    def __init__(self, 
                 model_name_or_path: str = "gradients-ai/fid_large_en_v1.0",
                 device: str = "cpu"):
        super().__init__()
        self.device = device
        print("Initializing model...")
        self.model = FiD.from_pretrained(model_name_or_path)
        self.model.to(self.device)
        print("Done!")
        print("Initializing tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        print("Done!")

    def append_question(
            self,
            question: str,
            documents: List[str],
            question_prefix: str = "Question: ",
            document_prefix: str = "Document: "
    ) -> List[str]:
        """Pair question to each document

        Args:
            question:
                a string - question
            documents:
                a list of string
        Returns:
            A question is paired with each document in `documents`
            become a list of string
        """

        if documents is None:
            return [question_prefix + question]
        return [question_prefix + question + " " + document_prefix + re.sub(r"_", " ", d) for d in documents] 

    def run(self, query, documents):
        # pprint(contexts)
        # print("Contexts len:", len(contexts))
        # print(top2_docs)

        inputs = self.append_question(
            query,
            list(doc.content for doc in documents)
        )
        print(inputs)
        tokenized_input = self.tokenizer(inputs, return_tensors="pt", padding=True)
        input_tensor = tokenized_input.input_ids[None, :, :].to(self.device)
        attention_mask = tokenized_input.attention_mask[None, :, :].to(self.device)
        print("Generating answers...")
        model_outputs = self.model.generate(
            input_ids=input_tensor,
            attention_mask=attention_mask,
            max_length=256,
            min_length=64,
            do_sample=True,
            num_beams=1,
            top_k=50,
            top_p=0.9,
            temperature=0.7,
            num_return_sequences=1,
            no_repeat_ngram_size=3,
            repetition_penalty=1.1
        )
        output = {"answer": []}
        print("Model output len:", len(model_outputs))
        for out in model_outputs:
            output["answer"].append(
                self.tokenizer.decode(out, skip_special_tokens=True)
            )
        return output, "output_1"
    
    def run_batch(self, queries: List[str], my_optional_param: Optional[int]):
        # process the inputs
        output = {"my_output": ...}
        return output, "output_1"

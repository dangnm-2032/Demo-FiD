from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import re

class WordSegmentor:
    def __init__(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained("NlpHUST/vi-word-segmentation")
        model = AutoModelForTokenClassification.from_pretrained("NlpHUST/vi-word-segmentation")
        self.pipeline = pipeline("token-classification", model=model, tokenizer=tokenizer)

    def segment(self, input):
        sentences = re.split(r"(,|;|\.|:|\"|\')", input)
        output = ""
        
        for sentence in sentences:
            ner_results = self.pipeline(sentence)
            example_tok = ""
            for e in ner_results:
                if "##" in e["word"]:
                    example_tok = example_tok + e["word"].replace("##","")
                elif e["entity"] =="I":
                    example_tok = example_tok + "_" + e["word"]
                else:
                    example_tok = example_tok + " " + e["word"]
            output += example_tok
            # print(len(output.split(" ")))

        return output

if __name__ == "__main__":
    segmentor = WordSegmentor()
    print(segmentor.segment("Xin chào, chúng tôi là sinh viên đại học Công nghệ thực phẩm. Ông Nguyễn Khắc Chúc  đang làm việc tại Đại học Quốc gia Hà Nội. Bà Lan, vợ ông Chúc, cũng làm việc tại đây."))
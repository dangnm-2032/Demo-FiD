from haystack.nodes.file_converter import PDFToTextConverter, TextConverter
from haystack.nodes import PreProcessor
from haystack.document_stores.faiss import FAISSDocumentStore
from haystack.nodes import DensePassageRetriever, EmbeddingRetriever
from haystack.pipelines import Pipeline
from haystack.schema import Document
from typing import Literal
import pandas as pd
import os
from time import sleep
from src.FiDReader import FiDReader
from src.Reranker import ReRanker
from src.WordSegmentor import WordSegmentor
import re

class GenerativeQA:
    def __init__(self,
                 file_path: str = None,
                 split_length: int = None,
                 model_name_or_path: str = "gradients-ai/fid_large_en_v1.0",
                 query_embedding_model: str = "facebook/dpr-question_encoder-si",
                 passage_embedding_model: str = "facebook/dpr-ctx_encoder-single-",
                 single_embedding_model: str = "BAAI/bge-large-en-v1.5",
                 rerank_model: str = 'llmrails/ember-v1',
                 retriever_option: Literal['dpr', 'eb'] | str = "dpr",
                 embedding_dim: int = 1024,
                 retriever_use_gpu: bool = False,
                 reranker_use_gpu: bool = False,
                 reader_use_gpu: bool = False,
                 valid_languages: Literal['en', 'vi'] | str = None,
                 meta: dict = {"company":"Company_1",
                               "processed": False}) -> None:
        
        if os.path.exists("faiss_document_store.db"):
            os.remove("faiss_document_store.db")
            sleep(1)
        
        self.ready = False
        self.converter = None
        self.meta = meta
        self.query_embedding_model = query_embedding_model
        self.passage_embedding_model = passage_embedding_model
        self.single_embedding_model = single_embedding_model
        self.retriever_option = retriever_option
        self.valid_languages = [valid_languages]
        self.retriever_use_gpu = retriever_use_gpu
        self.reranker_device = "cuda" if reranker_use_gpu else "cpu"
        self.reader_device = "cuda" if reader_use_gpu else "cpu"
        self.init_preprocessor(split_length)
        self.init_document_store(embedding_dim)
        self.init_reader_model(model_name_or_path)
        self.init_reranker(rerank_model)
        self.init_word_segmentor()
        
        if file_path:
            self.upload_document(file_path)

    def init_word_segmentor(self):
        if self.valid_languages[0] == 'vi':
            print("Init Word Segmentor...")
            self.word_segmentor = WordSegmentor()
            print("Done!")

    def init_reranker(self, model_name_or_path):
        print("Init Reranker...")
        self.reranker = ReRanker(
            model_name_or_path=model_name_or_path,
            device=self.reranker_device
        )
        print("Done!")

    def init_document_store(self, embedding_dim):
        print("Init Document Store...")
        self.document_store = FAISSDocumentStore(
            faiss_index_factory_str="Flat",
            embedding_dim=embedding_dim,
            return_embedding=True
        )
        print("Done!")

    def init_reader_model(self, model_name_or_path):
        print("Init Reader...")
        self.reader = FiDReader(
            model_name_or_path=model_name_or_path,
            device=self.reader_device
        )
        print("Done!")
    def set_valid_language(self, lang):
        if lang == "English":
            self.valid_languages = ['en']
        else:
            self.valid_languages = ['vi']
    
    def init_preprocessor(self, split_length):
        print("Init PreProcessor...")
        self.preprocessor = PreProcessor(
            clean_empty_lines=True,
    	    clean_whitespace=True,
    	    clean_header_footer=True,
	        split_respect_sentence_boundary=False,
            split_by='word',
            split_length=split_length,
            split_overlap=10
        )
        print("Done!")

    def query(self, question) -> None:
        print("Running pipeline...")
        # if self.valid_languages[0] == 'vi':
        #     question = self.word_segmentor.segment(question)
        answer = self.pipeline.run(query=question, debug=True)
        return answer

    def set_retriver_option(
            self, 
            option, 
            q_e_model,
            p_e_model,
            emb_model):
        self.retriever_option = option
        self.query_embedding_model = q_e_model
        self.passage_embedding_model = p_e_model
        self.single_embedding_model = emb_model

    def init_retriever(self) -> None:
        print("Init Retriever...")
        if self.retriever_option == 'dpr':
            self.retriever = DensePassageRetriever(
                document_store=self.document_store,
                use_gpu=self.retriever_use_gpu,
                query_embedding_model=self.query_embedding_model,
                passage_embedding_model=self.passage_embedding_model,
                top_k=20
            )
        else:
            self.retriever = EmbeddingRetriever(
                document_store=self.document_store,
                embedding_model=self.single_embedding_model,
                model_format="sentence_transformers",
                use_gpu=self.retriever_use_gpu,
                top_k=20
            )
        print("Done!")

    def init_pipeline(self) -> None:
        print("Init Pipeline...")
        self.pipeline = Pipeline()
        self.pipeline.add_node(component=self.retriever, name="retriever", inputs=["Query"])
        self.pipeline.add_node(component=self.reader, name="reader", inputs=["retriever"])
        # self.pipeline.add_node(component=self.reranker, name="reranker", inputs=["retriever"])
        # self.pipeline.add_node(component=self.reader, name="reader", inputs=["reranker"])
        print("Done!")
    
    def run_word_segment(self,
                         preprocessed) -> None:
        if self.valid_languages[0] != "vi":
            return preprocessed
        
        for id, doc in enumerate(preprocessed):
            preprocessed[id].content = self.word_segmentor.segment(doc.content)

        return preprocessed
    
    def upload_document(self,
                        file_path) -> None:
        converted = None
        print("Converting...")
        if ".pdf" in file_path.lower():
            self.converter = PDFToTextConverter(remove_numeric_tables=True, 
                                                valid_languages=self.valid_languages,)
            converted = self.converter.convert(file_path=file_path,
                                               meta=self.meta)
            preprocessed = self.preprocessor.process(converted)
            self.document_store.delete_documents()
            self.document_store.write_documents(preprocessed)
            self.init_retriever()
            self.document_store.update_embeddings(self.retriever,
                                                update_existing_embeddings=True)
        elif ".txt" in file_path.lower():
            self.converter = TextConverter(remove_numeric_tables=True, 
                                           valid_languages=self.valid_languages)
            converted = self.converter.convert(file_path=file_path,
                                               meta=self.meta)
            preprocessed = self.preprocessor.process(converted)
            self.document_store.delete_documents()
            self.document_store.write_documents(preprocessed)
            self.init_retriever()
            self.document_store.update_embeddings(self.retriever,
                                                update_existing_embeddings=True)
        elif ".json" in file_path.lower():
            df = pd.read_json(file_path, index_col=0)
            self.init_retriever()
            documents = []
            for idx, row in df.iterrows():
                title = row['title']
                content = row['content']
                embedding = self.retriever.embedding_encoder.embed(title)
                doc = Document(
                    content=content,
                    embedding=embedding
                )
                documents.append(doc)
            self.document_store.write_documents(documents=documents)

        elif ".csv" in file_path.lower():
            df = pd.read_csv(file_path, index_col=0)
            self.init_retriever()
            documents = []
            for idx, row in df.iterrows():
                title = row['title']
                content = row['content']
                embedding = self.retriever.embedding_encoder.embed(title)
                doc = Document(
                    content=content,
                    embedding=embedding
                )
                documents.append(doc)
            self.document_store.write_documents(documents=documents)
        print("Done!")
        self.init_pipeline()
        print("Ready!")
        self.ready = True

if __name__ == "__main__":
    model = GenerativeQA(
        file_path=None,
        split_length=200,
        model_name_or_path="gradients-ai/fid_large_en_v1.0",
        retriever_use_gpu=True,
        reranker_use_gpu=False,
        reader_use_gpu=False,
        embedding_dim=1024,
        valid_languages=['en'],
        retriever_option='eb',
        single_embedding_model="BAAI/bge-large-en-v1.5",
        query_embedding_model="thenlper/gte-large",
        passage_embedding_model="thenlper/gte-large",
    )

    pred = model.query("Who is the CEO?")
    print("Query:", pred['query'])
    print("Answer:", pred['answer'][0])

    while True:
        print("===============================")
        inp = input("Ask about the file: ")
        if "_exit" in inp:
            print("Bye bye")
            break

        pred = model.query(inp)
        print("Query:", pred['query'])
        print("Answer:", pred['answer'][0])

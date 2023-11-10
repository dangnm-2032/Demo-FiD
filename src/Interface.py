import gradio as gr
import time
from src.GenerativeQA import GenerativeQA
from src.voiceModel import AudioModel
import os

model = GenerativeQA(
    file_path=None,#"SDGs-2019_inglese.pdf",
    split_length=200,
    model_name_or_path="gradients-ai/fid_large_en_v1.0",
    retriever_use_gpu=False,
    reranker_use_gpu=False,
    reader_use_gpu=False,
    embedding_dim=1024,
    valid_languages=['en'],
    retriever_option='eb',
    single_embedding_model="BAAI/bge-large-en-v1.5",
    query_embedding_model="thenlper/gte-large",
    passage_embedding_model="thenlper/gte-large",
)

#audio_model = AudioModel(
#    s2t_model_name_or_path="facebook/wav2vec2-conformer-rel-pos-large-960h-ft",
#    t2s_model_name_or_path="suno/bark-small",
#    s2t_is_gpu=False,
#    t2s_is_gpu=False  
#)

def run():
    def vote(data: gr.LikeData):
        if data.liked:
            print("You upvoted this response: " + data.value)
        else:
            print("You downvoted this response: " + data.value)

    def add_file(history, file):
        global model
        # history = history + [((file.name,), None)]
        print(file.name)
        file_name = str(file.name).split("/")[-1]
        msg = f"'{file_name}' is uploaded!"
        history.append((msg, "new_file"))
        model.upload_document(file.name)
        return history

    def retriver_type_change(choice):
        # print("Hello")
        if choice == "Dense Passage Retriever":
            return [gr.update(visible=True), gr.update(visible=False)]
        if choice == "Embedding Retriever":
            return [gr.update(visible=False), gr.update(visible=True)]

    def apply_click(
        lang, 
        text_gen_model, 
        split_length,
        retriever_option,
        dpr_dim,
        q_e_model,
        p_e_model,
        eb_dim,
        emb_model
    ):
        global model
        print(
            "Apply settings:",
            lang, 
            split_length,
            retriever_option,
            text_gen_model,
            dpr_dim,
            q_e_model,
            p_e_model,
            eb_dim,
            emb_model,
            sep='\n - '
        )
        model.set_valid_language(lang)
        model.init_preprocessor(split_length)
        model.set_retriver_option(
            retriever_option,
            q_e_model,
            p_e_model,
            emb_model
        )
        model.init_reader_model(text_gen_model)

    with gr.Blocks() as demo:
        with gr.Tab("Chat"):
            welcome_msg = "Welcome to QA! ^^\nPlease upload a document (pdf, txt, csv, json) by below button to start."
            chatbot = gr.Chatbot([[None, welcome_msg]],
                                bubble_full_width=True, 
                                elem_id="chatbot",)
            with gr.Row():
                upload_doc = gr.UploadButton("🗀", 
                                            file_types=['.txt',
                                                        '.csv',
                                                        '.json',
                                                        '.pdf'],
                                            scale=1)
                msg = gr.Textbox(
                    show_label=False,
                    placeholder="Question goes in here...",
                    container=False,
                    show_copy_button=True,
                    scale=15
                )
                voice = gr.Audio(source='microphone', type='filepath')

            debug = gr.Textbox(
                label="Document Retriever",
                interactive=False
            )


            def user(user_message, history):
                print("User:", user_message)
                return "", history + [[user_message, None]]
            
            def user_voice(audio, history):
                global audio_model
                user_message = audio_model.s2t_transcribe(audio)
                print("User:", user_message)
                return history + [[user_message, None]]

            def bot(history, docs_retrieve):
                global model
                bot_message = ""
                docs_retrieve = ""
                # bot_message = random.choice(["How are you?", "I love you", "I'm very hungry"])
                if history[-1][1] == "new_file":
                    bot_message = "Ask me about the file :3"
                elif model.ready:
                    question = history[-1][0]
                    pred = model.query(question)
                    bot_message = pred['answer'][0]

                    for doc in pred['documents']:
                        # docs_retrieve += "Score: " + str(doc.score) + "\n"
                        docs_retrieve += doc.content
                        docs_retrieve += "\n====================================================\n"
                        docs_retrieve += "\n====================================================\n"
                        docs_retrieve += "\n====================================================\n"

                else:
                    bot_message = "Please upload a document for QA!"
                print("Bot:", bot_message)
                history[-1][1] = ""
                for character in bot_message:
                    history[-1][1] += character
                    time.sleep(0.01)
                    yield history, docs_retrieve

            def bot_voice(history):
                global audio_model
                text = history[-1][1]
                file = audio_model.t2s_transcribe(text)
                history += [(None, (file,))]
                return history

            msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
                bot, [chatbot, debug], [chatbot, debug]
            )
            chatbot.like(vote, None, None)
            # clear.click(lambda: None, None, chatbot, queue=False)
            file_msg = upload_doc.upload(
                add_file, 
                    [chatbot, upload_doc], 
                    [chatbot],
                    queue=False
            ).then(
                bot, 
                    [chatbot, debug], 
                    [chatbot, debug]
            )

            voice.stop_recording(
                user_voice,
                [voice, chatbot],
                chatbot
            ).then(
                bot, 
                [chatbot, debug], 
                [chatbot, debug]
            ).then(
                bot_voice,
                chatbot,
                chatbot
            )
        with gr.Tab("Settings"):
            with gr.Group():
                gr.Markdown(value="GENERAL",
                        container=False)
                lang_select = gr.Radio(
                    ["English", "Vietnamese"],
                    label="Language",
                    value="English",
                    interactive=True
                )
            with gr.Group():
                gr.Markdown(value="GENERATION MODEL",
                        container=False)
                text_gen_model = gr.Textbox(
                    value="gradients-ai/fid_large_en_v1.0",
                    label="FiD model name or path",
                    interactive=True
                )
            with gr.Group():
                gr.Markdown(value="DOCUMENT STORE",
                        container=False)
                split_length_inp = gr.Number(
                    value=200,
                    label="Split length",
                    info="Chunk",
                    interactive=True
                )
                retriever_option = gr.Radio(
                    ["Dense Passage Retriever", "Embedding Retriever"],
                    value="Dense Passage Retriever",
                    label="Retriever type",
                    interactive=True
                )
                with gr.Column():
                    with gr.Column(visible=True) as dprOption:
                        dpr_dim = gr.Number(
                            value=768,
                            label="Embedding dimension",
                            interactive=True
                        )
                        q_e_model = gr.Textbox(
                            value="facebook/dpr-question_encoder-si",
                            label="Query embedding model",
                            interactive=True
                        )
                        p_e_model = gr.Textbox(
                            value="facebook/dpr-ctx_encoder-single-",
                            label="Passage embedding model",
                            interactive=True
                        )
                    with gr.Column(visible=False) as ebOption:
                        eb_dim = gr.Number(
                            value=384,
                            label="Embedding dimension",
                            interactive=True
                        )
                        emb_model = gr.Textbox(
                            value="sentence-transformers/all-MiniLM-L6-v2",
                            label="Embedding model",
                            interactive=True
                        )
            with gr.Row():
                apply_button = gr.Button(
                    "Apply"
                )
                cancel_button = gr.Button(
                    "Cancel"
                )
            
            retriever_option.change(
                retriver_type_change, 
                retriever_option, 
                [dprOption, ebOption]
            )

            apply_button.click(
                apply_click,
                [lang_select,
                text_gen_model,
                split_length_inp,
                retriever_option,
                dpr_dim,
                q_e_model,
                p_e_model,
                eb_dim,
                emb_model]
            )
        
    demo.queue()
    try:
        demo.launch(server_name='0.0.0.0', share=True)
    except KeyboardInterrupt:
        tmp_path = "src/tmp"
        tmp_files = os.listdir(tmp_path)
        for file in tmp_files:
            if file != ".gitignore":
                os.remove(tmp_path + "/" + file)

from transformers import (
    Wav2Vec2ConformerForCTC, 
    Wav2Vec2Processor, 
    pipeline
)
import torch
import librosa
import numpy as np
import scipy
import os

class AudioModel:
    def __init__(
        self,
        s2t_model_name_or_path: str = "facebook/wav2vec2-conformer-rel-pos-large-960h-ft",
        t2s_model_name_or_path: str = "suno/bark-small",
        s2t_is_gpu: bool = False,
        t2s_is_gpu: bool = False    
    ) -> None:
        self.s2t_device = 'cuda' if s2t_is_gpu else "cpu"
        self.t2s_device = 'cuda' if t2s_is_gpu else "cpu"
        self.s2t_processor = Wav2Vec2Processor.from_pretrained(s2t_model_name_or_path)
        self.s2t_model = Wav2Vec2ConformerForCTC.from_pretrained(s2t_model_name_or_path).to(self.s2t_device)
        self.t2s_model = pipeline("text-to-speech", t2s_model_name_or_path, device=self.t2s_device)


    def s2t_transcribe(
        self, 
        audio_input: tuple
    ) -> str:
        sr, audio = audio_input
        audio = audio.astype(np.float32)
        audio /= np.max(np.abs(audio))

        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        input_values = self.s2t_processor(audio, sampling_rate=16000, return_tensors="pt").input_values
        logits = self.s2t_model(input_values.to(self.s2t_device)).logits
        pred_ids = torch.argmax(logits, dim=-1)
        pred_transcript = self.s2t_processor.batch_decode(pred_ids)[0]
        print(pred_transcript)
        return pred_transcript 
    
    def t2s_transcribe(
        self, 
        text: str,
        save_dir: str = "src/tmp"
    ) -> str:
        speech = self.t2s_model(text, forward_params={"do_sample": True})
        idx = len(os.listdir(save_dir))
        file_name = f"{save_dir}/voice_out_{idx}.wav"
        scipy.io.wavfile.write(file_name, rate=speech["sampling_rate"], data=speech["audio"][0])
        return file_name

if __name__ == "__main__":
    model = AudioModel()
    model.t2s_transcribe("The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's attention_mask to obtain reliable results.")
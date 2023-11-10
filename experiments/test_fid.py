from datasets import load_dataset

# msd_data = load_dataset("json", data_files=["data30_MSD.json"])

from accelerate.hooks import AlignDevicesHook, add_hook_to_module, attach_align_device_hook, attach_align_device_hook_on_blocks
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
    ModelOutput
)
from transformers.models.t5.modeling_t5 import T5_INPUTS_DOCSTRING, __HEAD_MASK_WARNING_MSG
import inspect
logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "T5Config"
from transformers import T5ForConditionalGeneration

class FiD(T5ForConditionalGeneration):
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, T5ForConditionalGeneration

        >>> tokenizer = AutoTokenizer.from_pretrained("t5-small")
        >>> model = T5ForConditionalGeneration.from_pretrained("t5-small")

        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits

        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""
        # print("attention_mask 1: ", attention_mask.size())
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # NOTE: FiD
        # Reshape from [batch, n_passages, length] to [batch * n_passages, length]
        if input_ids is not None:
            if input_ids.dim() == 3 and attention_mask.dim() == 3:
                #logger.info("Start Reshape from [batch, n_passage, length] to [batch * n_passage, length]")
                self.n_passages = input_ids.size(1)
                self.batch = input_ids.size(0)
                self.seq_length = input_ids.size(2)
                input_ids = input_ids.view(self.batch*self.n_passages, self.seq_length)
                attention_mask = attention_mask.view(self.batch*self.n_passages, self.seq_length)
            else:
                raise ValueError(f"NOT FiD TRAINING, got input_ids {input_ids.size()} and attention_mask {attention_mask.size()}")

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            # print("attention_mask 2: ", attention_mask.size())
            # print(f"batchsize: {self.batch}, n_passages: {self.n_passages}, seq_length: {self.seq_length}")
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # [batch*n_passages, seq_length, model_dim]
        hidden_states = encoder_outputs[0]

        # NOTE: FiD
        # from [batch*n_passages, seq_length, model_dim] to [batch, n_passages*seq_length, model_dim]
        hidden_states = hidden_states.view(self.batch, self.n_passages*self.seq_length, -1)

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # NOTE: change(FiD): reshape attention mask
        # print("attention_mask 3: ", attention_mask.size())
        # print(f"batchsize: {self.batch}, n_passages: {self.n_passages}, seq_length: {self.seq_length}")
        attention_mask = attention_mask.view(-1, self.n_passages*self.seq_length)
        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
    ) -> Dict[str, Any]:
        # print("attention_mask 4: ", model_kwargs["attention_mask"].size())
        if inputs_tensor.dim() == 3:
            #logger.info("Start Reshape from [batch, n_passage, length] to [batch * n_passage, length]")
            self.n_passages = inputs_tensor.size(1)
            self.batch = inputs_tensor.size(0)
            self.seq_length = inputs_tensor.size(2)
            inputs_tensor = inputs_tensor.view(self.batch*self.n_passages, self.seq_length)
            model_kwargs["attention_mask"] = model_kwargs["attention_mask"].view(self.batch*self.n_passages, self.seq_length)
            # print("attention_mask 5: ", model_kwargs["attention_mask"].size())
            # print(f"batchsize: {self.batch}, n_passages: {self.n_passages}, seq_length: {self.seq_length}")

        # 1. get encoder
        encoder = self.get_encoder()
        # Compatibility with Accelerate big model inference: we need the encoder to outputs stuff on the same device
        # as the inputs.
        if hasattr(self, "hf_device_map"):
            if hasattr(encoder, "_hf_hook"):
                encoder._hf_hook.io_same_device = True
            else:
                add_hook_to_module(encoder, AlignDevicesHook(io_same_device=True))

        # 2. Prepare encoder args and encoder kwargs from model kwargs.
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }
        encoder_signature = set(inspect.signature(encoder.forward).parameters)
        encoder_accepts_wildcard = "kwargs" in encoder_signature or "model_kwargs" in encoder_signature
        if not encoder_accepts_wildcard:
            encoder_kwargs = {
                argument: value for argument, value in encoder_kwargs.items() if argument in encoder_signature
            }

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name

        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        encoder_outputs = encoder(**encoder_kwargs)
        # print("encoder_outputs: ", encoder_outputs)
        encoder_outputs["last_hidden_state"] = encoder_outputs["last_hidden_state"].view(self.batch, self.n_passages*self.seq_length, -1)
        model_kwargs["encoder_outputs"]: ModelOutput = encoder_outputs
        # model_kwargs["attention_mask"] = model_kwargs["attention_mask"].view(self.batch, self.n_passages*self.seq_length)

        return model_kwargs

model = FiD.from_pretrained("gradients-ai/fid_large_vi_v1.0")
model.to("cpu")
tokenizer = AutoTokenizer.from_pretrained("gradients-ai/fid_large_vi_v1.0")

from typing import List, Optional, Tuple
def append_question(
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
    return [question_prefix + question + " " + document_prefix + d for d in documents]

docs = [
    """quá trình đạo hàm cho lan truyền ngược.
2.4.2.2. Quá trình huấn luyện
Để dùng mô hình này cho quá trình tiền huấn luyện, ta cần phải che đi một tỷ lệ các
time-step nhất định của feature encoder, tương tự như mask language modeling của
BERT [65]. Mục tiêu huấn luyện sẽ yêu cầu xác định chính xác vector lượng tử hóa
ẩn cho biểu diễn âm thanh trong một tập các bộ phân tâm cho mỗi time-step bị che.
Mô hình sau khi tiền huấn luyện có thể điều chỉnh trên dữ liệu có nhãn.
Masking hay che dữ liệu:  Để huấn luyện, mô hình sẽ phải dùng cơ chế mặt nạ che
đi một tỷ lệ đầu ra của feature encoder, hoặc nói cách khác là che đi các time-step
trước khi đưa nó vào mạng học ngữ cảnh (Transformer) và thay thế giá trị ở các
time-step này bằng cách feature vector được học và chia sẻ vector này cho tất cả các
time-step bị che, nhưng feature encoder sẽ không bị che khi đi qua mô-đun lượng tử
hóa. Để che đầu ra của feature encoder, mô hình sẽ ngẫu nhiên một tỷ lệ từ tất cả
các time-step để làm vị trí đầu tiên, và"""
]

inputs = append_question(
    "huấn luyện là gì?",
    docs
)
print(inputs)
tokenized_input = tokenizer(inputs, return_tensors="pt", padding=True)
input_tensor = tokenized_input.input_ids[None, :, :].to("cpu")
print(input_tensor.size())
attention_mask = tokenized_input.attention_mask[None, :, :].to("cpu")
print(attention_mask.size())

outputs = model.generate(
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
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(answer)

outputs_cs = model.generate(
    input_ids=input_tensor,
    attention_mask=attention_mask,
    penalty_alpha=0.6, top_k=32, max_length=512, use_cache=True
)
answer_cs = tokenizer.decode(outputs_cs[0], skip_special_tokens=True)
print(answer_cs)

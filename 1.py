import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBartTokenizer,
    default_data_collator,
    BertForTokenClassification,
    set_seed,
    MT5ForConditionalGeneration
)


path = "../../generation_event_extraction/models/score/XM_2_2021-09-22-06-45-16117_hfl_chinese-roberta-wwm-ext_run1"
constrant_config = AutoConfig.from_pretrained(
        path
    )
    
constrant_tokenizer = AutoTokenizer.from_pretrained(
        path
)
    
constrant_model = BertForTokenClassification.from_pretrained(
    path,
    config=constrant_config,
)


input = {"text":"[CLS]口腔来喝什么中药比较好，口腔癌能不能吃中药，哭相爱时应吃什么中药[SEP]口腔癌应该吃什么中药",
        "classification":0}
model_inputs = constrant_tokenizer(input["text"], padding=False, return_tensors="pt")

outputs = constrant_model(**model_inputs).logits
print(model_inputs)
print(outputs.shape)
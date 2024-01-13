import os
os.environ['TRANSFORMERS_CACHE'] = '/home/smoeding2/caches/'
os.environ['XDG_CACHE_HOME'] = '/home/smoeding2/caches/'
os.environ['WANDB_NOTEBOOK_NAME'] = 'LeoLM/leo-mistral-hessianai-7b'
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,HfArgumentParser,TrainingArguments,pipeline, logging
import bitsandbytes
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import os, torch, wandb
import datasets
from trl import SFTTrainer


base_model = "LeoLM/leo-mistral-hessianai-7b"

from accelerate import Accelerator

# Load base model(Mistral 7B)
bnb_config = BitsAndBytesConfig(  
    load_in_4bit= True,
 #   bnb_4bit_quant_type= "nf4",
 #   bnb_4bit_compute_dtype= torch.bfloat16,
#    bnb_4bit_use_double_quant= False,
)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
#   load_in_4bit=True,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",#{"": Accelerator().local_process_index},
    trust_remote_code=True,
)
#model.config.use_cache = False # silence the warnings. Please re-enable for inference!
#model.config.pretraining_tp = 1
#model.gradient_checkpointing_enable()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
tokenizer.add_bos_token, tokenizer.add_eos_token

#Adding the adapters in the layers
model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    lora_alpha=16,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    lora_dropout=0.4
#    target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]
)
#model = get_peft_model(model, peft_config)

training_arguments = TrainingArguments(
    output_dir="./result_full",
    per_device_train_batch_size=10,
    gradient_accumulation_steps=10,
    gradient_checkpointing=True,
    learning_rate=2.0e-5,
    logging_steps=1,
    num_train_epochs=1,
    max_steps=-1,
    report_to="wandb",
    save_steps=10,
    save_total_limit=10,
    bf16=False,
    lr_scheduler_type="constant",
    warmup_ratio=0.1,
    evaluation_strategy="epoch",
    logging_first_step=True,
)

import datasets
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/home/smoeding2/data/')

from elastic_search_dataset import ElasticSearchDataset
ds = ElasticSearchDataset(scroll_time='30d', scroll_size=100)

# Setting sft parameters
trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    max_seq_length= 1024,
    train_dataset=ds,
    dataset_text_field="text",
    peft_config=peft_config,
    tokenizer=tokenizer,
    packing= True,
)

trainer.train()

trainer.save_model("saved_models/trained")

wandb.finish()
model.config.use_cache = True
model.eval()
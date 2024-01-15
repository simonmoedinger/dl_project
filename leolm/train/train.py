import os
os.environ['TRANSFORMERS_CACHE'] = '/home/smoeding2/caches/'
os.environ['XDG_CACHE_HOME'] = '/home/smoeding2/caches/'
os.environ['WANDB_NOTEBOOK_NAME'] = 'LeoLM/leo-mistral-hessianai-7b'
import sys
from elastic_search_dataset import ElasticSearchDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training
import torch, wandb
from trl import SFTTrainer

base_model = "LeoLM/leo-mistral-hessianai-7b"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True
)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
tokenizer.add_bos_token, tokenizer.add_eos_token

# Adding the adapters in the layers
model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    lora_alpha=16,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    lora_dropout=0.4
)

training_arguments = TrainingArguments(
    output_dir="./result_full",
    per_device_train_batch_size=10,
    gradient_accumulation_steps=10,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    learning_rate=2.0e-5,
    logging_steps=1,
    num_train_epochs=1,
    weight_decay=0.001,
    max_steps=-1,
    report_to="wandb",
    save_steps=10,
    save_total_limit=10,
    bf16=False,  # Not available for too old Graphics cards
    #    lr_scheduler_type="cos",  # Chunk
    lr_scheduler_type="constant",  # Full
    warmup_ratio=0.1,
    evaluation_strategy="epoch",
    logging_first_step=True,
)

sys.path.insert(1, '/home/smoeding2/data/')

# Use this for chunk training
# ds = load_dataset("text", data_dir="./../data/dataset")["train"]

# Use this for full training
ds = ElasticSearchDataset(scroll_time='30d', scroll_size=100)

# Setting sft parameters
trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    max_seq_length=1024,
    train_dataset=ds,
    dataset_text_field="text",
    peft_config=peft_config,
    tokenizer=tokenizer,
    packing=True,
)

trainer.train()

trainer.save_model("saved_models/trained")

wandb.finish()
model.config.use_cache = True
model.eval()

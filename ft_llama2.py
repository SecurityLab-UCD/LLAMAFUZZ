import os
from transformers import AutoTokenizer, LlamaForCausalLM
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from accelerate import Accelerator, infer_auto_device_map, init_empty_weights
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_int8_training
from trl import SFTTrainer

# model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model_name = "meta-llama/Llama-2-7b-chat-hf"
new_model = "llama-2-7b-structured-jpg-20"

os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"


def formatting_prompts_func(examples):
    output_texts = []

    for i in range(len(examples["context"])):
        text = f"### Input: ```Generate a JPEG example in base64 encoded. Make sure the example is complete and valid. Only return the solution, no other words. Here is the text description you can refer A JPEG image is represented as a sequence of segments where each segment begins with a marker. Each marker starts with 0xFF byte followed by marker flag to represent the type of marker. The payload followed by marker is different as per marker type. Common JPEG marker types are as listed below:Short Name Bytes	Payload	Name Comments```\n```SOI	0xFF, 0xD8 none Start of Image```\n ```S0F0 0xFF, 0xC0	variable size Start of Frame``` \n```S0F2 0xFF, 0xC2	variable size Start fo Frame```\n```DHT	0xFF, 0xC4	variable size Define Huffman Tables```\n```DQT	0xFF, 0xDB	variable size Define Quantization Table(s)```\n```DRI	0xFF, 0xDD	4 bytes	Define Restart Interval```\n```SOS	0xFF, 0xDA	variable size	Start Of Scan```\n```RSTn 0xFF, 0xD//n//(//n//#0..7) none Restart```\n```APPn 0xFF, 0xE//n// variable size Application specific```\n```COM	0xFF, 0xFE variable size Comment```\n```EOI	0xFF, 0xD9	none End Of Image```\n```Within the entropy-coded data, after any 0xFF byte, a 0x00 byte is inserted by the encoder before the next byte, so that there does not appear to be a marker where none is intended, preventing framing errors. Decoders must skip this 0x00 byte. This technique, called byte stuffing (see JPEG specification section F.1.2.3), is only applied to the entropy-coded data, not to marker payload data. Note however that entropy-coded data has a few markers of its own; specifically the Reset markers (0xD0 through 0xD7), which are used to isolate independent chunks of entropy-coded data to allow parallel decoding, and encoders are free to insert these Reset markers at regular intervals (although not all encoders do this).```\n ### Output: {examples['context'][i]}"
        output_texts.append(text)

    return output_texts


# Load dataset (you can process it here)
dataset = load_dataset(
    "csv", data_files="../dataset/cleaneddata/jpg_base64.csv", split="train"
)

################################################################################
# QLoRA parameters
################################################################################

# Number of training epochs
num_train_epochs = 20

# LoRA attention dimension
lora_r = 64

# Alpha parameter for LoRA scaling
lora_alpha = 16

# Dropout probability for LoRA layers
lora_dropout = 0.1

################################################################################
# bitsandbytes parameters
################################################################################

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

################################################################################
# TrainingArguments parameters
################################################################################

# Output directory where the model predictions and checkpoints will be stored
output_dir = "./results"

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = True
bf16 = False

# Batch size per GPU for training
per_device_train_batch_size = 38

# Load the entire model on the GPU
device_map = "auto"  # {"": Accelerator().local_process_index}

# Batch size per GPU for evaluation
per_device_eval_batch_size = 4

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 1

# Enable gradient checkpointing
gradient_checkpointing = True

# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3

# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4

# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001

# Optimizer to use
optim = "paged_adamw_32bit"

# Learning rate schedule
lr_scheduler_type = "cosine"

# Number of training steps (overrides num_train_epochs)
max_steps = -1

# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True

# Save checkpoint every X updates steps
save_steps = 0

# Log every X updates steps
logging_steps = 25

################################################################################
# SFT parameters
################################################################################

# Maximum sequence length to use
max_seq_length = None

# Pack multiple short examples in the same input sequence to increase efficiency
packing = False


# Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)
# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)
# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    load_in_8bit=True,
    device_map=device_map,
)
# Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard",
    ddp_find_unused_parameters=False,
)

# accelerator = Accelerator()
# model, tokenizer, dataset = accelerator.prepare(model, tokenizer, dataset)
# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    formatting_func=formatting_prompts_func,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
)

# Train model
trainer.train()

# Save trained model
trainer.save_model(new_model)

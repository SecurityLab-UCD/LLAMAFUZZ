# Trianed from checkpoint
import os
from dataclasses import dataclass, field
from typing import Optional
import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
)

from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset

new_model = "llama-2-7b-structured-libjpg-hex"
dataset_path = "./prompts/fine-tune-libjpeg-mutate.csv"

device = Accelerator().local_process_index

def formatting_prompts_func(examples):
    print(examples)
    formated_prompt = f"### Input: ```Based on below libjpeg seed, mutate a new libjpeg seed. Make sure the example is complete and valid. Only return the solution, no other words. {examples['src']}```\n ### Output: {examples['cur']}"
    print(formated_prompt)
    return formated_prompt
    output_texts = []

    # f"### Input: ```Generate a seed for fuzzing libjpg in hex format. Make sure the example is complete and valid. Only return the solution, no other words.```\n ### Output: {examples['context']}"
    for i in range(len(examples["context"])):
        text = f"### Input: ```Generate a different PNG example in hex format. Make sure the example is complete and valid. Only return the solution, no other words.```\n ### Output: {examples['context'][i]}"
        output_texts.append(text)

    return output_texts

@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(
        default="meta-llama/Llama-2-7b-chat-hf", metadata={"help": "the model name"}
    )
    num_train_epochs: Optional[int] = field(
        default=50, metadata={"help": "Number of training epochs"}
    )
    per_device_train_batch_size: Optional[int] = field(
        default=1, metadata={"help": "the per device train batch size"}
    )
    seq_length: Optional[int] = field(
        default=1250, metadata={"help": "the sequence length"}
    )
    max_steps: Optional[int] = field(
        default=-1,
        metadata={"help": "Number of training steps (overrides num_train_epochs)"},
    )
    logging_steps: Optional[int] = field(
        default=25, metadata={"help": "the logging frequency"}
    )
    save_steps: Optional[int] = field(
        default=0, metadata={"help": "the saving frequency"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )
    group_by_length: Optional[bool] = field(
        default=False, metadata={"help": "whether to group by length"}
    )
    packing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use packing for SFTTrainer"}
    )
    lora_alpha: Optional[float] = field(
        default=16, metadata={"help": "the lora alpha parameter"}
    )
    lora_dropout: Optional[float] = field(
        default=0.05, metadata={"help": "the lora dropout parameter"}
    )
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})

    learning_rate: Optional[float] = field(
        default=2e-4, metadata={"help": "the learning rate"}
    )
    lr_scheduler_type: Optional[str] = field(
        default="cosine", metadata={"help": "the lr scheduler type"}
    )
    num_warmup_steps: Optional[int] = field(
        default=30, metadata={"help": "the number of warmup steps"}
    )
    weight_decay: Optional[float] = field(
        default=0.001, metadata={"help": "the weight decay"}
    )
    optimizer_type: Optional[str] = field(
        default="paged_adamw_32bit", metadata={"help": "the optimizer type"}
    )
    output_dir: Optional[str] = field(
        default="./results", metadata={"help": "the output directory"}
    )
    log_freq: Optional[int] = field(
        default=1, metadata={"help": "the logging frequency"}
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

peft_config = LoraConfig(
    r=script_args.lora_r,
    lora_alpha=script_args.lora_alpha,
    lora_dropout=script_args.lora_dropout,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

training_args = TrainingArguments(
    output_dir=script_args.output_dir,
    num_train_epochs=script_args.num_train_epochs,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    learning_rate=script_args.learning_rate,
    logging_steps=script_args.logging_steps,
    max_steps=script_args.max_steps,
    report_to="tensorboard",
    save_steps=script_args.save_steps,
    group_by_length=script_args.group_by_length,
    lr_scheduler_type=script_args.lr_scheduler_type,
    warmup_steps=script_args.num_warmup_steps,
    optim=script_args.optimizer_type,
    fp16=True,
    bf16=False,
    remove_unused_columns=False,
    run_name="sft_llama2",
    ddp_find_unused_parameters=False,
)


def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = formatting_prompts_func(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens


def create_datset(tokenizer, data_file_path):
    dataset = load_dataset(
        "csv",
        data_files=data_file_path,
        split="train",
        delimiter="/n",
    )
    chars_per_token = chars_token_ratio(dataset, tokenizer)
    dataset = ConstantLengthDataset(
        tokenizer,
        dataset,
        formatting_func=formatting_prompts_func,
        infinite=True,
        seq_length=script_args.seq_length,
        chars_per_token=chars_per_token,
    )
    return dataset


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    torch_dtype=torch.bfloat16,
    bnb_4bit_compute_dtype=torch.bfloat16,
    device_map=device,
)


tokenizer = AutoTokenizer.from_pretrained(
    script_args.model_name, trust_remote_code=True, padding=True
)
tokenizer.pad_token = tokenizer.bos_token
tokenizer.padding_side = "left" # Fix weird overflow issue with fp16 training
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

# Load dataset (you can process it here)
dataset = create_datset(tokenizer, dataset_path)

base_model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    quantization_config=bnb_config,
    device_map={"": Accelerator().local_process_index},
    trust_remote_code=True,
    use_auth_token=True,
)
base_model.config.use_cache = False




trainer = SFTTrainer(
    model=base_model,
    train_dataset=dataset,
    peft_config=peft_config,
    packing=script_args.packing,
    max_seq_length=script_args.seq_length,
    tokenizer=tokenizer,
    args=training_args,
)
trainer.train()  # resume_from_checkpoint=script_args.output_dir + "/final_checkpoint"

output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
trainer.save_model(new_model)
trainer.model.save_pretrained(output_dir)

# # Free memory for merging weights
# del base_model
# torch.cuda.empty_cache()

# model = AutoPeftModelForCausalLM.from_pretrained(
#     output_dir, device_map="auto", torch_dtype=torch.bfloat16
# )
# model = model.merge_and_unload()

# output_merged_dir = os.path.join(script_args.output_dir, "final_merged_checkpoint")
# model.save_pretrained(output_merged_dir, safe_serialization=True)

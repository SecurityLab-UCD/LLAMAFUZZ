# Trianed from checkpoint
import os
from dataclasses import dataclass, field
from typing import Optional
import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
)

from trl import SFTTrainer

target = "kamailio-parse_msg"
new_model = f"llama-2-7b-structured-{target}-mix-hex-mutator"
dataset_path = "/home/hxxzhang/dataset/csv/kamailio_fuzz_parse_msg.csv"

device = Accelerator().local_process_index


@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(
        default="meta-llama/Llama-2-7b-chat-hf", metadata={"help": "the model name"}
    )
    num_train_epochs: Optional[int] = field(
        default=20, metadata={"help": "Number of training epochs"}
    )
    per_device_train_batch_size: Optional[int] = field(
        default=1, metadata={"help": "the per device train batch size"}
    )
    seq_length: Optional[int] = field(
        default=1400, metadata={"help": "the sequence length"}
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
tokenizer.padding_side = "left"  # Fix weird overflow issue with fp16 training
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

# Load dataset (you can process it here)
dataset = load_dataset(
    "csv",
    data_files=dataset_path,
    split="train",
)

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
    dataset_text_field="context",
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

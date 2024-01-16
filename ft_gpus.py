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

based_model_name = "llama-2-7b-structured-jpg-hex-40"
new_model = "llama-2-7b-structured-jpg-png-hex"

device = Accelerator().local_process_index


def formatting_prompts_func_with_text(examples):
    output_texts = []

    for i in range(len(examples["context"])):
        text = f"### Input: ```Generate a different PNG example in hex format. Make sure the example is complete and valid. Only return the solution, no other words. Here is the text description you can refer The Examples Structure of a very simple PNG file 89 50 4E 47 0D 0A 1A 0A```\n```Contents of a minimal PNG file representing one red pixel Hex As characters 89 50 4E 47 0D 0A 1A 0A 00 00 00 0D 49 48 44 52```\n```00 00 00 01 00 00 00 01 08 02 00 00 00 90 77 53```\n```DE 00 00 00 0C 49 44 41 54 08 D7 63 F8 CF C0 00```\n```00 03 01 01 00 18 DD 8D B0 00 00 00 00 49 45 4E```\n```44 AE 42 60 82```\n```IHDR Chunk Offset into chunk	Hex Value	Decimal Value	Text	Meaning```\n```0x0D	13		IHDR chunk has 13 bytes of content```\n```0x49484452		IHDR	Identifies a Header chunk```\n```0x01	1		Image is 1 pixel wide```\n```0x01	1		Image is 1 pixel high```\n```0x08	8		8 bits per pixel (per channel)```\n```0x02	2		Color type 2 (RGB/truecolor)```\n```0x00	0		Compression method 0 (only accepted value)```0x00	0		Filter method 0 (only accepted value)```\n```0x00	0		Not interlaced```\n```0x907753DE			CRC of chunk's type and content (but not length)```\n```IDAT Chunk Offset into chunk	Hex Value	Meaning```\n```0x0C	IDAT chunk has 12 bytes of content```\n```0x49444154	Identifies a Data chunk```\n```0x08	DEFLATE compression method using a 256-byte window[44]```\n```0xD7	ZLIB FCHECK value, no dictionary used, maximum compression algorithm```\n```0x63F8CFC00000	A compressed DEFLATE block using the static Huffman code that decodes to 0x00 0xFF 0x00 0x00```\n```0x03010100	The ZLIB check value: the Adler32 checksum of the uncompressed data```\n```0x18DD8DB0	CRC of chunk's type and content (but not length)```\n ### Output: {examples['context'][i]}"
        output_texts.append(text)

    return output_texts


def formatting_prompts_func(examples):
    output_texts = []

    for i in range(len(examples["context"])):
        text = f"### Input: ```Generate a different PNG example in hex format. Make sure the example is complete and valid. Only return the solution, no other words.```\n ### Output: {examples['context'][i]}"
        output_texts.append(text)

    return output_texts


@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(
        default="meta-llama/Llama-2-7b-hf", metadata={"help": "the model name"}
    )
    num_train_epochs: Optional[int] = field(
        default=70, metadata={"help": "Number of training epochs"}
    )
    per_device_train_batch_size: Optional[int] = field(
        default=1, metadata={"help": "the per device train batch size"}
    )
    seq_length: Optional[int] = field(
        default=1900, metadata={"help": "the sequence length"}
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
        default=100, metadata={"help": "the number of warmup steps"}
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

if script_args.group_by_length and script_args.packing:
    raise ValueError("Cannot use both packing and group by length")

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


def formatting_prompts_func(examples):
    return f"### Input: ```Generate a different PNG example in hex format. Make sure the example is complete and valid. Only return the solution, no other words.```\n ### Output: {examples['context']}"
    output_texts = []

    for i in range(len(examples["context"])):
        text = f"### Input: ```Generate a different PNG example in hex format. Make sure the example is complete and valid. Only return the solution, no other words.```\n ### Output: {examples['context'][i]}"
        output_texts.append(text)

    return output_texts


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

base_model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    quantization_config=bnb_config,
    device_map={"": Accelerator().local_process_index},
    trust_remote_code=True,
    use_auth_token=True,
)
base_model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(
    script_args.model_name, trust_remote_code=True, padding=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

# Load dataset (you can process it here)
dataset = create_datset(tokenizer, "../dataset/cleaneddata/png_hex.csv")


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

# Free memory for merging weights
del base_model
torch.cuda.empty_cache()

model = AutoPeftModelForCausalLM.from_pretrained(
    output_dir, device_map="auto", torch_dtype=torch.bfloat16
)
model = model.merge_and_unload()

output_merged_dir = os.path.join(script_args.output_dir, "final_merged_checkpoint")
model.save_pretrained(output_merged_dir, safe_serialization=True)

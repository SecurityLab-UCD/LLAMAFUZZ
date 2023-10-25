# train from scratch
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
from trl import DPOTrainer

new_model = "llama-2-7b-structured-jpg-hex-40"


def formatting_prompts_func_with_text(examples):
    output_texts = []

    for i in range(len(examples["context"])):
        text = f"### Input: ```Generate a JPG example in hex format. Make sure the example is complete and valid. Only return the solution, no other words. Here is the text description you can refer A JPG image is represented as a sequence of segments where each segment begins with a marker. Each marker starts with 0xFF byte followed by marker flag to represent the type of marker. The payload followed by marker is different as per marker type. Common JPEG marker types are as listed below:Short Name Bytes	Payload	Name Comments```\n```SOI	0xFF, 0xD8 none Start of Image```\n ```S0F0 0xFF, 0xC0	variable size Start of Frame``` \n```S0F2 0xFF, 0xC2	variable size Start fo Frame```\n```DHT	0xFF, 0xC4	variable size Define Huffman Tables```\n```DQT	0xFF, 0xDB	variable size Define Quantization Table(s)```\n```DRI	0xFF, 0xDD	4 bytes	Define Restart Interval```\n```SOS	0xFF, 0xDA	variable size	Start Of Scan```\n```RSTn 0xFF, 0xD//n//(//n//#0..7) none Restart```\n```APPn 0xFF, 0xE//n// variable size Application specific```\n```COM	0xFF, 0xFE variable size Comment```\n```EOI	0xFF, 0xD9	none End Of Image```\n```Within the entropy-coded data, after any 0xFF byte, a 0x00 byte is inserted by the encoder before the next byte, so that there does not appear to be a marker where none is intended, preventing framing errors. Decoders must skip this 0x00 byte. This technique, called byte stuffing (see JPEG specification section F.1.2.3), is only applied to the entropy-coded data, not to marker payload data. Note however that entropy-coded data has a few markers of its own; specifically the Reset markers (0xD0 through 0xD7), which are used to isolate independent chunks of entropy-coded data to allow parallel decoding, and encoders are free to insert these Reset markers at regular intervals (although not all encoders do this).```\n ### Output: {examples['context'][i]}"
        output_texts.append(text)

    return output_texts


def formatting_prompts_func(examples):
    output_texts = []

    for i in range(len(examples["context"])):
        text = f"### Input: ```Generate a JPG example in hex format. Make sure the example is complete and valid. Only return the solution, no other words.```\n ### Output: {examples['context'][i]}"
        output_texts.append(text)

    return output_texts


@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(
        default="meta-llama/Llama-2-7b-chat-hf", metadata={"help": "the model name"}
    )
    num_train_epochs: Optional[int] = field(
        default=40, metadata={"help": "Number of training epochs"}
    )
    per_device_train_batch_size: Optional[int] = field(
        default=9, metadata={"help": "the per device train batch size"}
    )
    seq_length: Optional[int] = field(
        default=2048, metadata={"help": "the sequence length"}
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
        default=2, metadata={"help": "the gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )
    group_by_length: Optional[bool] = field(
        default=True, metadata={"help": "whether to group by length"}
    )
    packing: Optional[bool] = field(
        default=False, metadata={"help": "whether to use packing for SFTTrainer"}
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


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    if script_args.group_by_length and script_args.packing:
        raise ValueError("Cannot use both packing and group by length")

    base_model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map={"": Accelerator().local_process_index},
        trust_remote_code=True,
        use_auth_token=True,
    )
    base_model.config.use_cache = False

    peft_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "out_proj",
            "fc_in",
            "fc_out",
            "wte",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name, trust_remote_code=True, padding=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

    # Load dataset (you can process it here) NEED process
    dataset = load_dataset(
        "csv",
        data_files="../dataset/cleaneddata/jpg_hex.csv",
        split="train",
        delimiter="/n",
    )

    # 4. initialize training arguments:
    training_args = TrainingArguments(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        max_steps=script_args.max_steps,
        remove_unused_columns=False,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        learning_rate=script_args.learning_rate,
        # evaluation_strategy="steps",
        logging_first_step=True,
        logging_steps=script_args.logging_steps,  # match results in blog post
        output_dir=script_args.output_dir,
        optim="rmsprop",
        warmup_steps=150,
        report_to="tensorboard",
        bf16=False,
        fp16=True,
        run_name="dpo_llama2",
    )

    # 5. initialize the DPO trainer
    dpo_trainer = DPOTrainer(
        base_model,
        args=training_args,
        beta=script_args.beta,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_length=script_args.max_length,
        max_target_length=script_args.max_target_length,
        max_prompt_length=script_args.max_prompt_length,
        generate_during_eval=True,
    )

    # 6. train
    dpo_trainer.train()
    dpo_trainer.save_model(script_args.output_dir)

    # Free memory for merging weights
    torch.cuda.empty_cache()

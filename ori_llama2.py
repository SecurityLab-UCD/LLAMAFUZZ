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
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import time

model = LlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

prompts = [
    "Can you generate a HTTP response example? Only return the result, no other words or explinations.",
    "Can you generate a ELF example in base64 encoded format? Only return the result, no other words or explinations.",
    "Can you generate a Mach-O example in base64 encoded format? Only return the result, no other words or explinations.",
    "Can you generate a TTF example in base64 encoded format? Only return the result, no other words or explinations.",
    "Can you generate a OTF example in base64 encoded format? Only return the result, no other words or explinations.",
    "Can you generate a WOFF example in base64 encoded format? Only return the result, no other words or explinations.",
    "Can you generate a JSON example? Only return the result, no other words or explinations.",
    "Can you generate a ICC profile example in base64 encoded format? Only return the result, no other words or explinations.",
    "Can you generate a JPEG example in base64 encoded format? Only return the result, no other words or explinations.",
    "Can you generate a PCAP example in base64 encoded format? Only return the result, no other words or explinations.",
    "Can you generate a PNG example in base64 encoded format? Only return the result, no other words or explinations.",
    "Can you generate a XML example? Only return the result, no other words or explinations.",
    "Can you generate a DER certificate example in base64 encoded format? Only return the result, no other words or explinations.",
    "Can you generate a PHP example? Only return the result, no other words or explinations.",
    "Can you generate a OGG example in base64 encoded format? Only return the result, no other words or explinations.",
    "Can you generate a zip example in base64 encoded format? Only return the result, no other words or explinations.",
]
prompts = [
    "Can you generate a JPEG example in hex format? Make sure the example is complete and valid. Only return the result, no other words or explinations.",
    "Can you generate a JPEG example in hex format? Make sure the example is complete and valid. Only return the result, no other words or explinations.",
    "Can you generate a JPEG example in hex format? Make sure the example is complete and valid. Only return the result, no other words or explinations.",
    "Can you generate a JPEG example in hex format? Make sure the example is complete and valid. Only return the result, no other words or explinations.",
    "Can you generate a JPEG example in hex format? Make sure the example is complete and valid. Only return the result, no other words or explinations.",
    "Can you generate a JPEG example in hex format? Make sure the example is complete and valid. Only return the result, no other words or explinations.",
]

for prompt in prompts:
    start_time = time.time()
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    # Generate
    generate_ids = model.generate(
        input_ids, max_length=128, top_k=0, temperature=0.7
    )  #
    ans = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    end_time = time.time()
    # Calculate the elapsed time
    execution_time = end_time - start_time
    # Print the execution time
    print(f"Execution time: {execution_time} seconds")

    print("################START#HERE################")
    print(ans)
    print("@@@@@@@@@@@@@@@@END@HERE@@@@@@@@@@@@@@@@")

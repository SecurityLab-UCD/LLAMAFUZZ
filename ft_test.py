import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import PeftModel

new_model = "llama-2-7b-structured-libjpg-hex"
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, new_model)
model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained("./" + new_model)

prompts = [
    "### Input: Generate a PNG example. Make sure the example is complete and valid. Only return the solution, no other words.",
    "### Input: Generate a PNG example. Make sure the example is complete and valid. Only return the solution, no other words.",
    "### Input: Generate a PNG example. Make sure the example is complete and valid. Only return the solution, no other words.",
    "### Input: Generate a PNG example. Make sure the example is complete and valid. Only return the solution, no other words.",
    "### Input: Generate a PNG example. Make sure the example is complete and valid. Only return the solution, no other words.",
    "### Input: Generate a PNG example. Make sure the example is complete and valid. Only return the solution, no other words.",
]
for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt")

    input_ids = inputs.input_ids.to("cuda")
    # Generate
    generate_ids = model.generate(
        input_ids,
        max_length=1024,
        num_beams=5,
        no_repeat_ngram_size=2,
        num_return_sequences=5,
        early_stopping=True,
    )
    ans = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    print("################START#HERE################")
    print(ans)
    print("@@@@@@@@@@@@@@@@END@HERE@@@@@@@@@@@@@@@@")
    torch.cuda.empty_cache()

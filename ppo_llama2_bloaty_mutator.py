# PPO process that receive seeds from fuzzer then send back the mutated seeds to fuzzer
from dataclasses import dataclass, field
from typing import Optional
import os
import torch
import tyro
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig

from trl import (
    AutoModelForCausalLMWithValueHead,
    PPOConfig,
    PPOTrainer,
    set_seed,
)
import threading
import re
import sysv_ipc
import time
import struct

tqdm.pandas()

TYPE_SEED = 1
TYPE_EMPTY_SEED = 2
TYPE_REWARD = 3
TYPE_REQUEST = 4

access_token = "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"
cur_path = os.path.join(os.getcwd(), "structureLLM")
output_dir = os.path.join(cur_path, "ppo_checkpoint")
message_queue = []
seed_id_map = {}
id_rwd_map = {}
seeds_from_fuzzer = []
uid = 1
shared_resource_lock = threading.Lock()

@dataclass
class ScriptArguments:
    """
    Setup experiment config
    """
    ppo_config: PPOConfig = field(
        default_factory=lambda: PPOConfig(
            steps=10,
            model_name="llama-2-7b-structured-bloaty-hex-mutator",  # llama-2-7b-structured-jpg-hex-40
            query_dataset=None,
            reward_model=None,
            learning_rate=1e-5,
            log_with=None,
            mini_batch_size=1,
            batch_size=1,
            gradient_accumulation_steps=1,
            early_stopping=False,
            target_kl=0.1,
            kl_penalty="kl",
            ppo_epochs=6,
            seed=0,
            init_kl_coef=0.2,  # Initial KL coefficient.
            adap_kl_ctrl=True,  # Whether to adapt KL control.
            use_score_scaling=True,
            use_score_norm=True,
            score_clip=0.2,
            optimize_cuda_cache=True,
        )
    )
    peft_config: Optional[LoraConfig] = field(
        default_factory=lambda: LoraConfig(
            r=64,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            target_modules=[
                "q_proj",
                "down_proj",
                "gate_proj",
                "o_proj",
                "k_proj",
                "v_proj",
                "up_proj",
            ],
            task_type="CAUSAL_LM",
        ),
    )
    trust_remote_code: bool = field(
        default=True, metadata={"help": "Enable `trust_remote_code`"}
    )


args = tyro.cli(ScriptArguments)

def mq_thread():
    """
    Thread to receive request from fuzzer, and send generated seed to fuzzer
    """
    global message_queue, seed_id_map, seeds_from_fuzzer
    try:
        mq = sysv_ipc.MessageQueue(1234, sysv_ipc.IPC_CREAT)
    except sysv_ipc.ExistentialError:
        print(f"Message queue with key {1234} already exists.")
        return
    while True:
        # only receive request msg
        msg, mtype = mq.receive(type=TYPE_REQUEST)
        seeds_from_fuzzer.append(msg.decode())
        if not message_queue == []:
            # send uid + seed
            seed = message_queue.pop(0)
            mq.send(
                struct.pack("I", seed_id_map[seed]) + seed.encode("utf-8"),
                True,
                type=TYPE_SEED,
            )
        else:
            # send empty str do default muatation
            mq.send("", True, type=TYPE_EMPTY_SEED)


def reward_thread():
    """
    Thread to receive reward info from fuzzer. Reward info stored in global id_rwd_map.
    """
    try:
        # Create a new message queue or get an existing one
        rw_mq = sysv_ipc.MessageQueue(4321, sysv_ipc.IPC_CREAT)
    except sysv_ipc.ExistentialError:
        print(f"Message queue with key {4321} already exists.")
        return
    while True:
        rw_msg, rw_mtype = rw_mq.receive(type=TYPE_REWARD)
        # receive reward msg(uid + reward)
        decoded_msg = struct.unpack("ii", rw_msg)
        global id_rwd_map
        if decoded_msg[0] in id_rwd_map:
            # reward should be in range [-3,3]
            if decoded_msg[1] > 4000:
                id_rwd_map[decoded_msg[0]] = 3.0
            else:
                id_rwd_map[decoded_msg[0]] = float(decoded_msg[1]) / 1000.0 - 3.0
        else:
            rw_mq.send(
                struct.pack("I", decoded_msg[0]) + struct.pack("I", decoded_msg[1]),
                True,
                type=TYPE_REWARD,
            )


def calculate_reward(seed_batch):
    """
    Calculate rewards by bitmap.

    Returns:
        list of rewards.
    """
    global seed_id_map, id_rwd_map
    start_time = time.time()
    while time.time() - start_time < 100:
        if id_rwd_map[seed_id_map[seed_batch[0]]] != 0.0:
            return [torch.tensor(id_rwd_map[seed_id_map[i]]) for i in seed_batch]
    return [torch.tensor(id_rwd_map[seed_id_map[i]]) for i in seed_batch]


def hex_string_to_hex(hex_string):
    """
    Formatting generated hex string.

    Returns:
        String of hex.
    """
    print(hex_string)
    if len(hex_string.split("### Output:"))>=2:
        hex_string = hex_string.split("### Output:")[1]
        
    hex_string = re.sub(r"[^a-zA-Z0-9\s]", " ", hex_string)
    hex_values = hex_string.replace("0x", " ")

    sections = hex_values.split()  # Split the string into sections

    # Iterate through the sections and add leading zeros if needed
    result = []
    for section in sections:
        if len(section) == 1:
            section = "0" + section
            result.append(section)
        elif len(section) == 2:
            result.append(section)
    return "".join(result)



def main():
    """
    Main function to run PPO loop
    """
    # Init the tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(cur_path, args.ppo_config.model_name),
        use_fast=True,
        token=access_token,
    )
    # Some tokenizers like GPT-2's don't have a padding token by default, so we set one here.
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.bos_token
    # tokenizer.padding_side = "left"
    # We retrieve the dataloader by calling the `build_dataset` function.

    # set seed before initializing value head for deterministic eval
    set_seed(args.ppo_config.seed)

    # Build the model.
    peft_config = args.peft_config
    ref_model = None
    # Copy the model to each device
    current_device = Accelerator().local_process_index
    device_map = {"": current_device}

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        os.path.join(cur_path, args.ppo_config.model_name),
        trust_remote_code=args.trust_remote_code,
        device_map=device_map,
        peft_config=peft_config,
        load_in_4bit=True,
        token=access_token,
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        ),
        # use_flash_attention_2=True, Unable to use this feature in current GPU
    )
    # Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    
    # flash attention 1
    torch.backends.cuda.sdp_kernel(
        enable_flash=True, enable_math=False, enable_mem_efficient=False
    )
    default_prompt = "### Input: ```Based on below hex bloaty seed, mutate a new bloaty seed. Make sure the example is complete and valid. Only return the solution. 0x00x61,0x730x6d,0x00x41,0x760x43,0x20x57,0x1d0x1,0x10x1,0xe80x3,0x10xff,0xff0x0,0x00x1,0x00x3,0x4b0x0,0x00x0,0x10x1,0x10x1,0x1d0x1,0x10x1,0x10x1,0x10x1,0x10x1,0x10x1,0x10x1,0x10x1,0x10x11,0x10xcc,0xcc0xcc,0xcc0xcc,0xcc0xcc,0xcc0xcc,0xcc0xcc,0xcc0xcc,0xcc0xcc,0xcc0xcc,0xcc0xcc,0xcc0xcc,0xcc0xcc,0xcc0xcc,0xcc0xcc,0xcc0xcc,0xcc0x1,0x10x1,0x10x1,0x10x1,0x10x1,0x10x1,0x10x1,0x10x1,0x10x1,0x10x1,0x10x1,0x10x1,0x10x1```"

    generation_kwargs = {
        "do_sample": True,
        "min_length": -1,
        "top_p": 0.85, # 0.9
        # "top_k": 1250,
        "pad_token_id": tokenizer.bos_token_id,
    }

    while True:
        global seeds_from_fuzzer
        if seeds_from_fuzzer != []:
            prompt = "### Input: ```Based on below hex bloaty seed, mutate a new bloaty seed. Make sure the example is complete and valid. "+seeds_from_fuzzer.pop()+"```"
        else:
            prompt = default_prompt
        query_tensors = tokenizer(prompt, return_tensors="pt")["input_ids"].to('cuda')
        
        response_tensors = model.generate(
            input_ids=query_tensors,
            max_length=1000, # Input_length is less than 700, the max_length should be longer than Input_length, but may lead to slow generation
            **generation_kwargs,
        )


        response = tokenizer.batch_decode(
            response_tensors, skip_special_tokens=True
        )

        # Compute sentiment score
        global uid, seed_id_map, id_rwd_map, message_queue
        seed_batch = []
        for r in response:
            # set default rewards to 0
            seed = hex_string_to_hex(r)
            shared_resource_lock.acquire()
            seed_id_map[seed] = uid + os.getpid()
            id_rwd_map[uid + os.getpid()] = float(0.0)
            shared_resource_lock.release()
            message_queue.append(seed)
            seed_batch.append(seed)
            print("seed:::",seed)
        uid += 8
        # iterate msgs record reward
        rewards = calculate_reward(seed_batch)
        print("Rewards:::", rewards)

        torch.cuda.empty_cache()
    ppo_trainer.save_model("ppo-llama2-jpg")
    ppo_trainer.save_pretrained(output_dir)


if __name__ == "__main__":
    t = threading.Thread(
        target=mq_thread,
        args=(),
    )
    t.start()
    # if accelerator.is_main_process:
    t2 = threading.Thread(target=reward_thread, args=())
    t2.start()
    main()

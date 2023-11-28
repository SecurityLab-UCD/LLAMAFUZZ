from dataclasses import dataclass, field
from typing import Optional
import os
import torch
import tyro
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline, HfArgumentParser, BitsAndBytesConfig

from trl import (
    AutoModelForCausalLMWithValueHead,
    PPOConfig,
    PPOTrainer,
    set_seed,
)
from trl.core import LengthSampler
from PIL import Image

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

output_dir = "./result"
message_queue = []
seed_id_map = {}
id_rwd_map = {}
uid = 1
shared_resource_lock = threading.Lock()


def mq_thread():
    global message_queue, seed_id_map
    try:
        mq = sysv_ipc.MessageQueue(1234, sysv_ipc.IPC_CREAT)
    except sysv_ipc.ExistentialError:
        print(f"Message queue with key {1234} already exists.")
        return
    while True:
        # only receive request msg
        msg, mtype = mq.receive(type=TYPE_REQUEST)
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
    global id_rwd_map
    try:
        # Create a new message queue or get an existing one
        rw_mq = sysv_ipc.MessageQueue(4321, sysv_ipc.IPC_CREAT)
    except sysv_ipc.ExistentialError:
        print(f"Message queue with key {4321} already exists.")
        return
    while True:
        # receive msg
        rw_msg, rw_mtype = rw_mq.receive(type=TYPE_REWARD)
        # receive reward msg(uid + reward)
        decoded_msg = struct.unpack("ii", rw_msg)
        print("RECEIVE seedid", decoded_msg[0], " REWARDS", decoded_msg[1])
        id_rwd_map[decoded_msg[0]] = float(decoded_msg[1])


def hex_string_to_hex(hex_string):
    hex_string = hex_string.replace(
        "Generate a JPG example in hex format. Make sure the example is complete and valid. Only return the solution, no other words.",
        " ",
    )
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

    # Join the sections back together with spaces
    return "".join(result)


@dataclass
class ScriptArguments:
    ppo_config: PPOConfig = field(
        default_factory=lambda: PPOConfig(
            steps=10,
            model_name="llama-2-7b-structured-jpg-hex-40",
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
            ppo_epochs=4,
            seed=0,
            init_kl_coef=0.2,  # Initial KL coefficient.
            adap_kl_ctrl=True,  # Whether to adapt KL control.
            use_score_scaling=False,
            use_score_norm=False,
            score_clip=None,
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


# Build the dataset.
def build_dataset(tokenizer, input_min_text_length=2, input_max_text_length=16):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    # load imdb with datasets
    ds = load_dataset(
        "csv",
        data_files="../dataset/cleaneddata/jpg_question.csv",
        split="train",
        delimiter="/n",
    )

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["context"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def main():
    # Init the tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(
        "./" + args.ppo_config.model_name, use_fast=True
    )
    # Some tokenizers like GPT-2's don't have a padding token by default, so we set one here.
    tokenizer.pad_token_id = tokenizer.eos_token_id
    # We retrieve the dataloader by calling the `build_dataset` function.
    dataset = build_dataset(tokenizer)

    # set seed before initializing value head for deterministic eval
    set_seed(args.ppo_config.seed)

    # Build the model.
    peft_config = args.peft_config
    ref_model = None
    # Copy the model to each device
    device_map = {"": Accelerator().local_process_index}

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        "./" + args.ppo_config.model_name,
        trust_remote_code=args.trust_remote_code,
        device_map=device_map,
        peft_config=peft_config,
        load_in_4bit=True,
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
    model.gradient_checkpointing_enable()

    # We then build the PPOTrainer, passing the model, the reference model, the tokenizer
    ppo_trainer = PPOTrainer(
        args.ppo_config,
        model,
        ref_model,
        tokenizer,
        dataset=dataset,
        data_collator=collator,
    )
    # Define the arguments to pass to the `generate` function.
    generation_kwargs = {
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
    }
    # flash attention 1
    torch.backends.cuda.sdp_kernel(
        enable_flash=True, enable_math=False, enable_mem_efficient=False
    )

    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        query_tensors = batch["input_ids"]
        ppo_trainer.accelerator.unwrap_model(model).gradient_checkpointing_disable()

        response_tensors = ppo_trainer.generate(
            query_tensors,
            return_prompt=True,
            length_sampler=LengthSampler(2, 1028),
            **generation_kwargs,
        )
        print("end generate")

        ppo_trainer.accelerator.unwrap_model(model).gradient_checkpointing_enable()

        batch["response"] = tokenizer.batch_decode(
            response_tensors, skip_special_tokens=True
        )
        # batch["ref_response"] = tokenizer.batch_decode(ref_response_tensors)

        # Compute sentiment score
        global uid, seed_id_map, id_rwd_map, message_queue
        seed_batch = []
        # i = 1
        for r in batch["response"]:
            # set default rewards to 0
            seed = hex_string_to_hex(r)
            shared_resource_lock.acquire()
            # i += 1
            print("SEED ", uid + os.getpid(), " :")
            seed_id_map[seed] = uid + os.getpid()
            id_rwd_map[uid + os.getpid()] = 0.0
            shared_resource_lock.release()
            message_queue.append(seed)
            seed_batch.append(seed)
        uid += 8
        # iterate msgs record reward
        # need wait for 0.1s for response?
        time.sleep(1.1)
        rewards = [torch.tensor(id_rwd_map[seed_id_map[i]]) for i in seed_batch]
        print(rewards)

        # Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)
        print("Step " + str(epoch) + " finished")
        torch.cuda.empty_cache()

    ppo_trainer.save_pretrained(output_dir)


if __name__ == "__main__":
    t = threading.Thread(
        target=mq_thread,
        args=(),
    )
    t2 = threading.Thread(target=reward_thread, args=())
    t.start()
    t2.start()
    main()

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
import csv
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
uid = 1
shared_resource_lock = threading.Lock()

def hex_string_to_hex(hex_string):
    """
    Formatting generated hex string.

    Returns:
        String of hex.
    """
    hex_string = hex_string.replace(
        "### Output:",
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
    return "".join(result)

def mq_thread():
    """
    Thread to receive request from fuzzer, and send generated seed to fuzzer
    """
    global message_queue, seed_id_map
    with open('./prompts/mock_mutator.csv',newline='\n') as file:
        reader = csv.reader(file)
        for row in reader:
            message_queue.append(hex_string_to_hex(row))
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
if __name__ == "__main__":
    t = threading.Thread(
        target=mq_thread,
        args=(),
    )
    t.start()
    # if accelerator.is_main_process:
    t2 = threading.Thread(target=reward_thread, args=())
    t2.start()
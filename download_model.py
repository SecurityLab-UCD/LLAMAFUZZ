# PPO process that receive seeds from fuzzer then send back the mutated seeds to fuzzer
from dataclasses import dataclass, field
from typing import Optional
import os
import torch
import tyro
from accelerate import Accelerator
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig

from trl import (
    AutoModelForCausalLMWithValueHead,
    PPOConfig,
    set_seed,
)
import threading
import re
import sysv_ipc
import struct
import random

tqdm.pandas()

TYPE_SEED = 1
TYPE_TEXT_SEED = 2
TYPE_REWARD = 3
TYPE_REQUEST = 4

access_token = "YOUR ACCESS TOKEN"
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    token=access_token,
)
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    token=access_token,
)
exit()

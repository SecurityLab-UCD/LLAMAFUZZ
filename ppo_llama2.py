from dataclasses import dataclass, field
from typing import Optional

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

from threading import Thread
from hashlib import sha256
import sysv_ipc
import time

tqdm.pandas()

TYPE_REQUEST = 1
TYPE_REWARD = 2
TYPE_SEED = 3
TYPE_EMPTY_SEED = 4

model_name = "meta-llama/Llama-2-7b-chat-hf"
output_dir = "./result"
message_queue = []
hashmap = {}


def mq_thread():
    global message_queue, hashmap
    mq = sysv_ipc.MessageQueue(1234, sysv_ipc.IPC_CREAT)
    while True:
        # receive msg
        msg, mtype = mq.receive()
        if mtype == TYPE_REQUEST:
            print("RECEIVE REQUEST")
            if not message_queue == []:
                # send seed
                print("SEND SEED queue", message_queue)
                mq.send(message_queue.pop(0), True, type=TYPE_SEED)
            else:
                print("SEND EMPTY SEED")
                # send empty str do default muatation
                mq.send("", True, type=TYPE_EMPTY_SEED)
        # it is reward msg
        elif mtype == TYPE_REWARD:
            decoded_msg = msg.decode("utf-8")
            hashmap[sha256(decoded_msg[0]).hexdigest()] = decoded_msg[1]


def hex_string_to_hex(hex_string):
    hex_values = hex_string.replace(",", " ").replace("0x", " ")

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
    return " ".join(result)


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

    t = Thread(
        target=mq_thread,
        args=(),
    )
    t.start()

    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        query_tensors = batch["input_ids"]
        print("Start generate")

        ppo_trainer.accelerator.unwrap_model(model).gradient_checkpointing_disable()

        response_tensors = ppo_trainer.generate(
            query_tensors,
            return_prompt=True,
            length_sampler=LengthSampler(2, 1028),
            **generation_kwargs,
        )
        print("End generate")

        ppo_trainer.accelerator.unwrap_model(model).gradient_checkpointing_enable()

        batch["response"] = tokenizer.batch_decode(
            response_tensors, skip_special_tokens=True
        )
        # batch["ref_response"] = tokenizer.batch_decode(ref_response_tensors)

        # Compute sentiment score
        for r in batch["response"]:
            # set default rewards to 0
            hashmap[sha256(r.encode("utf-8")).hexdigest()] = 0.0
            message_queue.append(r)
            print("MESSAGE:::")
            print(r, "\nresponse appende to mq")

        # iterate msgs record reward
        # need wait for 0.1s for response?
        time.sleep(0.1)
        rewards = [
            torch.tensor(hashmap[sha256(i.encode("utf-8")).hexdigest()])
            for i in batch["response"]
        ]
        print(rewards)

        # Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)
        print("Step " + str(epoch) + " finished")
        torch.cuda.empty_cache()

    ppo_trainer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()

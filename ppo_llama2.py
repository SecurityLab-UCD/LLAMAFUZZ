from dataclasses import dataclass, field
from typing import Optional

import torch
import tyro
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline, HfArgumentParser

from trl import (
    AutoModelForCausalLMWithValueHead,
    PPOConfig,
    PPOTrainer,
    set_seed,
)
from trl.core import LengthSampler
from PIL import Image

tqdm.pandas()

model_name = "meta-llama/Llama-2-7b-chat-hf"
output_dir = "./result"


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
            model_name="meta-llama/Llama-2-7b-chat-hf",
            query_dataset=None,
            reward_model=None,
            learning_rate=1.41e-5,
            log_with=None,
            mini_batch_size=1,
            batch_size=2,
            gradient_accumulation_steps=1,
            early_stopping=False,
            target_kl=6.0,
            kl_penalty="kl",
            seed=0,
            fp16=True,
            bf16=False,
            use_score_scaling=False,
            use_score_norm=False,
            score_clip=None,
        )
    )
    peft_config: Optional[LoraConfig] = field(
        default_factory=lambda: LoraConfig(
            r=8,
            lora_alpha=16,
            bias="none",
            target_modules=[
                "q_proj",
                "v_proj",
                "k_proj",
                "out_proj",
                "fc_in",
                "fc_out",
                "wte",
            ],
            task_type="CAUSAL_LM",
        ),
    )
    trust_remote_code: bool = field(
        default=True, metadata={"help": "Enable `trust_remote_code`"}
    )


args = tyro.cli(ScriptArguments)


# Build the dataset.
def build_dataset(tokenizer, input_min_text_length=2, input_max_text_length=8):
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

    # ds = ds.rename_columns({"text": "review"})
    # ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

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
    tokenizer = AutoTokenizer.from_pretrained(args.ppo_config.model_name)
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
        args.ppo_config.model_name,
        trust_remote_code=args.trust_remote_code,
        device_map=device_map,
        peft_config=peft_config,
        load_in_4bit=True,
    )
    # Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # We then build the PPOTrainer, passing the model, the reference model, the tokenizer
    ppo_trainer = PPOTrainer(
        args.ppo_config,
        model,
        ref_model,
        tokenizer,
        dataset=dataset,
        data_collator=collator,
    )
    # We then define the arguments to pass to the `generate` function. These arguments
    # are passed to the `generate` function of the PPOTrainer, which is a wrapper around
    # the `generate` function of the trained model.
    generation_kwargs = {
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
    }

    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        query_tensors = batch["input_ids"]

        response_tensors = ppo_trainer.generate(
            query_tensors,
            return_prompt=False,
            # generate_ref_response=False,
            **generation_kwargs,
        )
        batch["response"] = tokenizer.batch_decode(response_tensors)
        # batch["ref_response"] = tokenizer.batch_decode(ref_response_tensors)

        # Compute sentiment score
        # texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        # pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
        # rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]

        # rewards = []
        # for r in zip(batch["response"]):
        #     # Todo:: seeds rename
        #     try:
        #         with open("./seeds/" + str(epoch), "wb") as file:
        #             file.write(bytes.fromhex(hex_string_to_hex(r[0])))
        #         Image.open("./seeds/" + str(epoch))
        #         rewards.append(1.0)
        #     except Exception as e:
        #         rewards.append(-1.0)

        # rewards = [torch.tensor(reward) for reward in rewards]

        rewards = [torch.tensor(1.0) for i in batch["response"]]

        print(rewards)
        # ref_texts = [q + r for q, r in zip(batch["query"], batch["ref_response"])]
        # ref_pipe_outputs = sentiment_pipe(ref_texts, **sent_kwargs)
        # ref_rewards = [torch.tensor(output[1]["score"]) for output in ref_pipe_outputs]

        # batch["ref_rewards"] = ref_rewards

        # Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)
        print("Step " + str(epoch) + " finished")

    ppo_trainer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()

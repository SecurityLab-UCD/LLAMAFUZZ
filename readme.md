# LLAMAFUZZ
Large Language Model for mutating structured data.

This repository contains a collection of tools and models for mutating structured data using large language models (LLMs). The key components of the repository include fine-tuned checkpoint models, scripts for fine-tuning, and tools for generating structured seeds to augment the fuzzing process.

## Repository Structure

- 'ft_gpu-mutator.py': Script used for fine-tuning the model.
- 'llama2-mutator.py': Script used for generating structured seeds to augment the fuzzing process. It builds asynchronous communication with the fuzzer, receives seeds from the fuzzer, performs mutations, and sends the mutated seeds back to the fuzzer.
- 'example/': Directory containing example datasets.
- 'README.md': This file, providing an overview of the repository and instructions for usage.
## Branches

main: Includes the fine-tuned checkpoint model for the Magma experiment.
fuzzbench: Includes the fine-tuned checkpoint model for the Fuzzbench experiment.
## Dependencies

To use the tools and models in this repository, you need to install the dependencies listed in 'requirement.txt'

## Fine-Tuning the Model

To fine-tune the model using the provided script, use the following command:

python ft_gpu-mutator.py
## Running the Llama2 Mutator

To run the llama2-mutator.py script and start the process of generating structured seeds for fuzzing, use the following command:

'accelerate launch --mixed_precision fp16 structureLLM/llama2_mutator.py --fuzzing_target <fuzzing_target> --fuzzing_object <fuzzing_object> --if_text <True/False>'

Replace <fuzzing_target> and <fuzzing_object> with the appropriate targets and objects for your fuzzing process.

if_text specifies whether the model processes text-based files or the binary-based files.

## Example Usage

Example datasets are provided in the /example directory to help you get started. You can use these datasets to test the fine-tuning and mutating processes.
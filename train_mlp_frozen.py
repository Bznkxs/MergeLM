# flake8: noqa
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
# regular:
accelerate launch train_mlp_frozen.py     \
    --model_name_or_path="../models/Mistral-7B-v0.1" \
    --dataset_name="/gpfs/fs1/home/du/project/datasets/slimpajama_per_source_downsample/slimpajama_packed_65535_5b_per_source_down_sample_0.1.hf" \
    --dataset_text_field="text"    \
    --report_to="wandb"     \
    --attn_implementation="flash_attention_2"     \
    --torch_dtype="bfloat16"     \
    --bf16="True"     \
    --learning_rate=2e-5     \
    --lr_scheduler_type="constant"     \
    --per_device_train_batch_size=1     \
    --gradient_accumulation_steps=32     \
    --output_dir="../models/Mistral-7B-64k"     \
    --logging_steps=1     \
    --num_train_epochs=1     \
    --max_steps=-1     \
    --gradient_checkpointing  \
    --save_steps=1   \
    --save_total_limit=20   \
    --dataloader_num_workers=4

# peft:
python examples/scripts/sft.py \
    --model_name_or_path="facebook/opt-350m" \
    --dataset_text_field="text" \
    --report_to="wandb" \
    --learning_rate=1.41e-5 \
    --per_device_train_batch_size=64 \
    --gradient_accumulation_steps=16 \
    --output_dir="sft_openassistant-guanaco" \
    --logging_steps=1 \
    --num_train_epochs=3 \
    --max_steps=-1 \
    --push_to_hub \
    --gradient_checkpointing \
    --gradient_checkpointing_kwargs={'use_reentrant':False} \
    --use_peft \
    --lora_r=64 \
    --lora_alpha=16
"""

import logging
import os
from contextlib import nullcontext

from trl.commands.cli_utils import init_zero_verbose, SFTScriptArguments, TrlParser
from trl.env_utils import strtobool
from transformers import AutoTokenizer, AutoModelForCausalLM

TRL_USE_RICH = strtobool(os.getenv("TRL_USE_RICH", "0"))

if TRL_USE_RICH:
    init_zero_verbose()
    FORMAT = "%(message)s"

    from rich.console import Console
    from rich.logging import RichHandler

import torch
from datasets import load_dataset, load_from_disk

from tqdm.rich import tqdm
from transformers import AutoTokenizer

from trl import (
    ModelConfig,
    RichProgressCallback,
    SFTConfig,
    SFTTrainer,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map,
)

tqdm.pandas()

if TRL_USE_RICH:
    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)


if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    args, training_args, model_config = parser.parse_args_and_config()

    # Force use our print callback
    if TRL_USE_RICH:
        training_args.disable_tqdm = True
        console = Console()

    ################
    # Model init kwargs & Tokenizer
    ################
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = None  # model_kwargs
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    raw_datasets = load_from_disk(args.dataset_name)

    train_dataset = raw_datasets  # [args.dataset_train_split]

    # chunk the dataset, so that each chunk has 2^16 tokens. Split each datapoint into two chunks.
    # def chunk_2_16_tokens(examples):
    #     source = examples["input_ids"]
    #     source_len = len(source)
    #     chunks = []
    #     for sentence in source:
    #         chunks += [sentence[:2 ** 15], sentence[2 ** 15:]]
    #     return {"input_ids": chunks}
    #
    # train_dataset = train_dataset.map(
    #     chunk_2_16_tokens,
    #     batched=True,
    #     num_proc=100
    # )

    # eval_dataset = raw_datasets[args.dataset_test_split]

    ################
    # Optional rich context managers
    ###############
    init_context = nullcontext() if not TRL_USE_RICH else console.status("[bold green]Initializing the SFTTrainer...")
    save_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status(f"[bold green]Training completed! Saving the model to {training_args.output_dir}")
    )


    ################
    # load model
    ################

    modified_model_kwargs = {k:v for k,v in model_kwargs.items()}
    modified_model_kwargs["torch_dtype"] = torch.float16 if modified_model_kwargs["torch_dtype"] == "float16" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **modified_model_kwargs)
    # change model config rope theta
    model.config.rope_theta = 1000000

    # Freeze all mlp layers, io layers, and embeddings
    # for name, param in model.named_parameters():
    #
    #     if "mlp" in name or "lm_head" in name or "embed_tokens" in name:
    #         param.requires_grad = False
    #         print(name, ": frozen")
    #     else:
    #         param.requires_grad = True
    #         print(name, ": not frozen")

    ################
    # Training
    ################
    with init_context:
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
    #         eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            peft_config=get_peft_config(model_config),
            callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
            max_seq_length=65536//4,
        )

    trainer.train()

    with save_context:
        trainer.save_model(training_args.output_dir)
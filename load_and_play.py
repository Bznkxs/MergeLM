import argparse
import sys
import os
import shutil
import logging
import time
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
# sentencepiece, protobuf
# from model_merging_methods.merging_methods import MergingMethod
import torch
import numpy as np
import random
# from model_merging_methods.task_vector import TaskVector
# from utils.utils import get_param_names_to_merge, get_modules_to_merge
# from utils.read_retrieval_head import read_retrieval_head
# from model_merging_methods.mask_weights_utils import mask_model_weights, mask_input_with_mask
# from utils.test_model import test_model_completion, prepare_model_from_cpu, load_model_from_checkpoint

random.seed(13245)
# copy the dependencies
def copy_params_to_model(params, model):  # copying code from model_merging_methods.merging_methods
    for param_name, param_value in model.named_parameters():
        if param_name in params:
            param_value.data.copy_(params[param_name])
        else:
            print(f"param_name {param_name} not in params")

def read_retrieval_head(retrieval_head_file="./Mistral-7B-v0.1.json", cutoff=0.1, *args, **kwargs):
    # retrieval_head_file = "../Mistral-7B-v0.1.json"
    with open(retrieval_head_file, "r") as f:
        head_list = json.load(f)
    head_score_list = [([int(number) for number in item[0].split('-')], np.mean(item[1])) for item in head_list.items()]
    head_score_list = sorted(head_score_list, key=lambda x: x[1], reverse=True)
    if kwargs.get("random") is not None:
        head_score_list = random.sample(head_score_list, kwargs.get("random"))
    else:
        i = 0
        for i, (head, score) in enumerate(head_score_list):
            if score < cutoff:
                print(f"{i} of {len(head_score_list)} heads ({i/len(head_score_list)}) have score at least {cutoff}")
                return head_score_list[:i]

    return head_score_list

def smart_tokenizer_and_embedding_resize(
        special_tokens_dict,
        tokenizer,
        model,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    assert tokenizer.vocab_size == 32000
    print("Original vocab size: 32000")
    print(tokenizer.special_tokens_map)
    print(special_tokens_dict)
    tokenizer.add_special_tokens({"additional_special_tokens": []})  # a bug in huggingface tokenizers
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)

    print(f"Added {num_new_tokens} new tokens to the tokenizer.")
    # print all the special tokens
    print(tokenizer.special_tokens_map)
    if num_new_tokens > 0:
        model.resize_token_embeddings(tokenizer.vocab_size + num_new_tokens)

        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def mask_input_with_mask(input_tensor: torch.Tensor, mask: torch.Tensor, use_rescale: bool, mask_rate=None):
    """
    mask the input with a given mask
    mask: same shape as input_tensor, 1 for the parameters that we want to keep, 0 for the parameters that we want to mask
    mask_rate: float. If none, calculate the mask rate as the ratio of the number of parameters that we want to keep to the total number of parameters
    """
    masked_input_tensor = input_tensor * mask
    if use_rescale:
        if mask_rate is None:
            mask_rate = torch.sum(mask) / torch.numel(mask)
            mask_rate = 1 - mask_rate.item()

        if mask_rate < 1.0:

            masked_input_tensor = torch.div(input=masked_input_tensor, other=1 - mask_rate)
    return masked_input_tensor

def test_model_completion(enc, model, prompt):
    prompt = enc(prompt, return_tensors="pt")
    input_ids = prompt["input_ids"].to(model.device)
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=1000)
        response = enc.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
    return response

def diff(model1, model2):
    for name1, param1 in model1.named_parameters():
        param2 = model2.state_dict()[name1]
        if not torch.equal(param1, param2):
            print(name1, "different")

######################################
# Main code
######################################

save_path = sys.argv[1]

base_enc = AutoTokenizer.from_pretrained(save_path)
base_model = AutoModelForCausalLM.from_pretrained(save_path)

base_model=base_model.to("cuda")
print(test_model_completion(base_enc, base_model, "The capital of the United States is") )
print(test_model_completion(base_enc, base_model, "[User] Write a python code that solves the problem: given a list of numbers, find the maximum and minimum elements. Your code should be a function that takes a list as the only argument, and return a tuple, which consists of four numbers: the maximum value, the index where the first maximum value appears, the minimum value, and the index where the first minimum value appears. [Assistant]"))


while True:
    prompt = input("Enter a prompt: ")
    print(test_model_completion(base_enc, base_model, prompt))
# base_model=base_model.to("cuda:1")
# print(test_model_completion(base_enc, base_model, "[User] Write a python code that solves the problem: given a list of numbers, find the maximum and minimum elements. Your code should be a function that takes a list as the only argument, and return a tuple, which consists of four numbers: the maximum value, the index where the first maximum value appears, the minimum value, and the index where the first minimum value appears. [Assistant]"))
#
# model1=model1.to("cuda:2")
# print(test_model_completion(enc1, model1, "[User] Write a python code that solves the problem: given a list of numbers, find the maximum and minimum elements. Your code should be a function that takes a list as the only argument, and return a tuple, which consists of four numbers: the maximum value, the index where the first maximum value appears, the minimum value, and the index where the first minimum value appears. [Assistant]"))
# model2=model2.to("cuda:3")
# print(test_model_completion(enc2, model2, "[User] Write a python code that solves the problem: given a list of numbers, find the maximum and minimum elements. Your code should be a function that takes a list as the only argument, and return a tuple, which consists of four numbers: the maximum value, the index where the first maximum value appears, the minimum value, and the index where the first minimum value appears. [Assistant]"))
#


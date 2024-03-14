import sys
import os
import json
import random
import copy
import shutil
import argparse
from typing import List

import numpy as np
from datasets import Dataset, Features
from datasets import Image as ImageFeature
from datasets import Value


DATASET_NAME = 'interior-design-prompt-editing-dataset'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="interior-design-all")
    args = parser.parse_args()
    return args

def get_prompt(directory: str):
    prompt_file = os.path.join(directory, 'prompt.json')
    prompt_json = read_json(prompt_file)
    return prompt_json['edit']

def read_json(filename):
    with open(filename, 'r') as file:
        return json.loads(file.read())

def get_paths(root: str):
    data = []
    for directory in os.listdir(os.path.join(root)):
        if not os.path.isdir(os.path.join(root, directory)):
            continue
        for file in os.listdir(os.path.join(root, directory)):
            if not file.endswith("_0.jpg"):
                continue
            file_name = os.path.join(root, directory, file)
            data.append((file_name, 
                get_prompt(os.path.join(root, directory)), 
                file_name.replace("_0.jpg", "_1.jpg"))) 
    return data

def generate_examples(data_paths):
    def fn():
        for data_path in data_paths:
            yield {
                "original_image": {"path": data_path[0]},
                "edit_prompt": data_path[1],
                "designed_image": {"path": data_path[2]},
            }
    return fn

def main(args):
    data_paths = get_paths(args.data_root)
    data_paths = data_paths[:int(0.8*len(data_paths))]
    generation_fn = generate_examples(data_paths)
    print("Creating dataset...")
    ds = Dataset.from_generator(
        generation_fn,
        features=Features(
            original_image=ImageFeature(),
            edit_prompt=Value("string"),
            designed_image=ImageFeature(),
        ),
    )

    print("Pushing to the Hub...")
    ds.push_to_hub(DATASET_NAME)


if __name__ == "__main__":
    args = parse_args()
    main(args)
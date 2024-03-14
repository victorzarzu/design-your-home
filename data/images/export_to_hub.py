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


DATASET_NAME_TRAIN = 'interior-design-prompt-editing-dataset-train'
DATASET_NAME_TEST = 'interior-design-prompt-editing-dataset-test'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="interior-design-all")
    args = parser.parse_args()
    return args

def get_prompt(directory: str, type: str):
    prompt_file = os.path.join(directory, 'prompt.json')
    prompt_json = read_json(prompt_file)
    return prompt_json[type]

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
                get_prompt(os.path.join(root, directory), 'input'), 
                get_prompt(os.path.join(root, directory), 'edit'), 
                get_prompt(os.path.join(root, directory), 'output'),
                file_name.replace("_0.jpg", "_1.jpg"))) 
    return data

def generate_examples(data_paths):
    def fn():
        for data_path in data_paths:
            yield {
                "original_image": {"path": data_path[0]},
                "input_prompt": data_path[1],
                "edit_prompt": data_path[2],
                "output_prompt": data_path[3],
                "designed_image": {"path": data_path[4]},
            }
    return fn

def main(args):
    data_paths = get_paths(args.data_root)
    random.shuffle(data_paths)
    data_paths_train = data_paths[:int(0.8*len(data_paths))]
    data_paths_test = data_paths[int(0.8*len(data_paths)):]

    unchanged_paths = get_paths('./unchanged-data')
    unchanged_paths = unchanged_paths[:int(0.1*len(unchanged_paths))]

    data_paths_test = data_paths_test + unchanged_paths
    generation_fn_train = generate_examples(data_paths_train)
    generation_fn_test = generate_examples(data_paths_test)
    print("Creating dataset...")
    ds_train = Dataset.from_generator(
        generation_fn_train,
        features=Features(
            original_image=ImageFeature(),
            input_prompt=Value("string"),
            edit_prompt=Value("string"),
            output_prompt=Value("string"),
            designed_image=ImageFeature(),
        ),
    )
    ds_test = Dataset.from_generator(
        generation_fn_test,
        features=Features(
            original_image=ImageFeature(),
            input_prompt=Value("string"),
            edit_prompt=Value("string"),
            output_prompt=Value("string"),
            designed_image=ImageFeature(),
        ),
    )

    print("Pushing to the Hub...")
    ds_train.push_to_hub(DATASET_NAME_TRAIN)
    ds_test.push_to_hub(DATASET_NAME_TEST)


if __name__ == "__main__":
    args = parse_args()
    main(args)
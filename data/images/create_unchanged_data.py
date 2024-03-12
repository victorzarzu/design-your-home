
import sys
import os
import json
import random
import copy
import shutil

objects = dict({'living room': set(['sink', 'bath', 'toilet', 'iron']),
                'bathroom': set(['bed', 'couch', 'pillow', 'oven', 'gas cooker']),
                'kitchen': set(['couch', 'bath', 'toilet', 'iron', 'bed']),
                'bedroom': set(['iron', 'oven', 'gas cooker', 'sink'])})

if len(sys.argv) < 3:
    print("Error in arguments!")
    exit()

data_directory = sys.argv[1]
dest_directory = sys.argv[2]
select_prob = 0.05

def read_json(filename):
    with open(filename, 'r') as file:
        return json.loads(file.read())

def save_json(filename, prompt_json):
    with open(filename, 'w') as output_file:
        json.dump(prompt_json, output_file, indent=2)

def create_duplicate(directory, prompt_json, output_dir, object):
    os.makedirs(output_dir)
    output_prompt_json = copy.deepcopy(prompt_json)
    output_prompt_json['output'] = output_prompt_json['input']
    output_prompt_json['edit'] = f'Remove the {object}.'.format(object = object)
    save_json(os.path.join(output_dir, 'prompt.json'), output_prompt_json)

    for file in os.listdir(directory):
        if file.endswith("_0.jpg"):
            source_image_path = os.path.join(directory, file)
            destination_image_path_input = os.path.join(output_dir, file)
            destination_image_path_output = os.path.join(
                output_dir, file.replace("_0.jpg", "_1.jpg")
            )

            shutil.copy(source_image_path, destination_image_path_input)
            shutil.copy(source_image_path, destination_image_path_output)

def get_directories(directory):
    directories = []
    for file in os.listdir(directory):
        if os.path.isdir(os.path.join(data_directory, file)):
            directories.append(os.path.join(data_directory, file))
    return directories

def augment_data(directories, dest_directory):
    nextDir = min(len(os.listdir(dest_directory)), 1)
    print(directories)
    for directory in directories:
        if len(os.listdir(directory)) <= 1:
            continue
        prompt_file = os.path.join(directory, 'prompt.json')
        prompt_json = read_json(prompt_file)
        if 'living room' in prompt_json['input']:
            diff = objects['living room']
        elif 'bathroom' in prompt_json['input']:
            diff = objects['bathroom']
        elif 'bedroom' in prompt_json['input']:
            diff = objects['bedroom']
        elif 'kitchen' in prompt_json['input']:
            diff = objects['kitchen']
        else:
            continue

        for object in diff:
            if random.random() <= select_prob:
                nextDir_str = str(nextDir).zfill(7) 
                create_duplicate(directory = directory, prompt_json = prompt_json, output_dir = os.path.join(dest_directory, nextDir_str), object = object)
                nextDir += 1

augment_data(get_directories(directory = data_directory), dest_directory=dest_directory)
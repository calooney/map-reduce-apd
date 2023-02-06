import os
import re
import chardet
import json
import shutil
import numpy as np

ENGLISH_LETTERS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

def print_usage():
    print("[ERROR] bad usage")
    print("usage:   mpiexec -n [processes number] python3 main.py [input_dir] [output_dir")
    print("example: mpiexec -n 9 python3 main.py test-files output_dir")


def cleanup_dump(dirpath):
    if os.path.exists(dirpath):
        shutil.rmtree(dirpath)

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def asign_mappers_workload(input_dir, mapper_nodes):
    file_names = os.listdir(input_dir)
    total_size = 0
    file_size = {}

    for file_name in file_names:
        file_path = input_dir + "/" +  file_name
        current_file_size = os.path.getsize(file_path)
        total_size += current_file_size
        file_size[file_name] = current_file_size

    target_workload = total_size / len(mapper_nodes)

    response = {}
    mapper_load = {}
    for mapper in mapper_nodes:
        mapper_load[mapper] = 0
        response[mapper] = []

    sorted_file_size = sorted(file_size.items(), key=lambda item: item[1], reverse=True)
    for fname, fsize in sorted_file_size:
        mapper = min(mapper_load.items(), key=lambda x: x[1])[0]
        mapper_load[mapper] += fsize
        response[mapper].append(fname)

    return response


def read_file(filename):
    with open(filename, 'rb') as f:
        result = chardet.detect(f.read())
        encoding = result['encoding']
    with open(filename, 'r', encoding=encoding) as f:
        text = f.read()
    # process the text as desired
    return text

def compute_word_map(file_path):
    word_map = {}
    contents = read_file(file_path)
    words = re.findall(r'\b\w+\b', contents)
    for word in words:
        word = word.lower()
        if word in word_map:
            word_map[word] += 1
        else:
            word_map[word] = 1

    return word_map

def dump_dict(data, file_path):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)

def dump_string(data, file_path):
    with open(file_path, "a") as file:
        file.write(data)
    
def load_dict(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def invert_dict(word_map, file_name):
    result = {}
    for word, count in word_map.items():
        if word not in result:
            result[word] = {file_name: count}
        else:
            result[word][file_name] += count
    return result

def merge_dict(a, b):
    result = {**a, **b}
    for key, value in result.items():
        if key in a and key in b:
            result[key] = {**a[key], **b[key]}
    return result

def get_reducer_letters(rank, reducers):
    index = reducers.index(rank)
    letter_chunks = np.array_split(ENGLISH_LETTERS, len(reducers))
    
    return letter_chunks[index]
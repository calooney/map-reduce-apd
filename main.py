#/bin/python3
# mpiexec --oversubscribe -n <PROC_NUMBER> python3 main.py <INPUT_DIR> <OUTPUT_DIR>

from mpi4py import MPI
from utils import *
import sys
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
COMM_TAG = 99

# CONFIG
MASTER_NODE   = 0
MAPPER_NODES  = [1, 2, 3, 4]
REDUCER_NODES = [5, 6]

# PRECALCULATION
ROOT_DIR = os.path.abspath(os.getcwd()) + '/'
DUMP_DIR = ROOT_DIR + "dump"
MAPPING_DUMP_DIR  = DUMP_DIR + "/mapper_input_dataset"
MAPPER_RESULT_DIR = DUMP_DIR + "/mapper_result_dump"
REDUCER_DATA_DIR  = DUMP_DIR + "/reducer_input_dataset"
RESULT_DUMP_DIR   = DUMP_DIR + "/result_dump"

DUMPS_DIRS = [ MAPPING_DUMP_DIR, MAPPER_RESULT_DIR, REDUCER_DATA_DIR, RESULT_DUMP_DIR ]

if (len(sys.argv) != 3):
    if (rank == 1):
        print_usage()
else:
    input_dir  = ROOT_DIR + sys.argv[1]
    output_dir = ROOT_DIR + sys.argv[2]
    
    if rank is MASTER_NODE:
        # INIT
        cleanup_dump(DUMP_DIR)

        for directory in DUMPS_DIRS:
            create_directory(directory)

        # ASIGN MAPPER WORKLOAD
        input_file_names = os.listdir(input_dir)
        planned_workload = asign_mappers_workload(input_dir, MAPPER_NODES)

        for mapper, workload in planned_workload.items():
            comm.send(workload, dest=mapper, tag=COMM_TAG)

        # GET MAPPER RESULTS
        mapper_results_dirs = []
        for mapper in MAPPER_NODES:
            response = comm.recv(source=MPI.ANY_SOURCE, tag=COMM_TAG)
            mapper_results_dirs.append(response)

        # ASIGN REDUCER WORKLOAD
        
        for reducer in REDUCER_NODES:
            to_do_letters = get_reducer_letters(reducer, REDUCER_NODES)
            comm.send(to_do_letters, reducer, tag=COMM_TAG)
            comm.send(mapper_results_dirs, reducer, tag=COMM_TAG)

        # WAIT FOR PROCESSING
        for reducer in REDUCER_NODES:
            comm.recv(source=MPI.ANY_SOURCE, tag=COMM_TAG)

        print ("ALL GOOD!")
        print ("Doamne ajuta!")

    elif rank in MAPPER_NODES:
        response = {}
        files = comm.recv(source=MASTER_NODE, tag=COMM_TAG)
        dump_string(str(rank) + " -> " + str(files) + "\n", MAPPING_DUMP_DIR + "/mappers_workload.log")
        
        for file_name in files:
            file_path = input_dir + "/" + file_name
            word_map = compute_word_map(file_path)

            name, ext = os.path.splitext(file_name)
            dump_dict(word_map, MAPPING_DUMP_DIR + "/" + name + "_word-map.json")
            
            response = merge_dict(response, invert_dict(word_map, file_name))
        
        result_dump_path = MAPPER_RESULT_DIR + "/" + str(rank) + "_all_words.json"
        dump_dict(response, result_dump_path)

        current_mapper_dir = MAPPER_RESULT_DIR + "/" + str(rank)
        create_directory(current_mapper_dir)

        letter_dict = {}
        for word, dict in response.items():
            key_letter = word[0]
            current_word_item = {word: dict}

            if (key_letter in letter_dict):
                letter_dict[key_letter] = merge_dict(letter_dict[key_letter], current_word_item)
            else:
                letter_dict[key_letter] = current_word_item
        
        for letter in ENGLISH_LETTERS:
            dump_dict(letter_dict[letter], current_mapper_dir + "/" + letter + ".json")

        comm.send(current_mapper_dir, dest=MASTER_NODE, tag=COMM_TAG)

    elif rank in REDUCER_NODES:
        to_do_letters = comm.recv(source=MPI.ANY_SOURCE, tag=COMM_TAG)
        log_msg = str(rank) + " -> " + str(to_do_letters) + "\n"
        dump_string(log_msg, REDUCER_DATA_DIR + "/reducers_letters.log")

        mappers_result_dirs = comm.recv(source=MPI.ANY_SOURCE, tag=COMM_TAG)
        working_dict = {}
        for mapper_dir in mappers_result_dirs:
            for letter in to_do_letters:
                target_letter_path = mapper_dir + "/" + letter + ".json"
                working_dict = merge_dict(working_dict, load_dict(target_letter_path))
        
        dump_dict(working_dict, REDUCER_DATA_DIR + "/reducer_" + str(rank) + "_dataset.json")     

        results = {}
        for word, files in working_dict.items():
            if any(word.startswith(letter) for letter in to_do_letters):
                result_files = dict(sorted(files.items(), key=lambda item: item[1], reverse=True))
                results[word] = result_files
        
        dump_dict(results, RESULT_DUMP_DIR + "/results_" + str(rank) + ".json")

        comm.send(True, MASTER_NODE, tag=COMM_TAG)
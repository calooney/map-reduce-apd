#/bin/python3
# mpiexec --oversubscribe -n $PROC_NUMBER python3 main.py <input_dir> <output_dir>

from mpi4py import MPI
from utils import *
import sys
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
COMM_TAG = 99

# CONFIG
MASTER_NODE   = 1
MAPPER_NODES  = [2, 3, 4]
REDUCER_NODES = [5, 6]

#PRECALCULATION
ROOT_DIR = os.path.abspath(os.getcwd()) + '/'
DUMP_DIR = ROOT_DIR + "dump"
MAPPING_DUMP_DIR  = DUMP_DIR + "/input_mapping_dump"
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

        for dir in DUMPS_DIRS:
            if not os.path.exists(dir):
                os.makedirs(dir)

        # ASIGN MAPPER WORKLOAD
        input_file_names = os.listdir(input_dir)
        planned_workload = asign_mappers_workload(input_dir, MAPPER_NODES)

        for mapper, workload in planned_workload.items():
            comm.send(workload, dest=mapper, tag=COMM_TAG)

        # GET MAPPER RESULTS
        files = []
        for mapper in MAPPER_NODES:
            result_file = comm.recv(source=MPI.ANY_SOURCE, tag=COMM_TAG)
            files.append(result_file)

        # ASIGN REDUCER WORKLOAD
        for reducer in REDUCER_NODES:
            comm.send(files, reducer, tag=COMM_TAG)

        # WAIT FOR PROCESSING
        for reducer in REDUCER_NODES:
            comm.recv(source=MPI.ANY_SOURCE, tag=COMM_TAG)

        print ("ALL GOOD!")
        print ("Doamne ajuta!")

    elif rank in MAPPER_NODES:
        response = {}
        files = comm.recv(source=MASTER_NODE, tag=COMM_TAG)
        for file_name in files:
            file_path = input_dir + "/" + file_name
            word_map = compute_word_map(file_path)

            name, ext = os.path.splitext(file_name)
            dump_dict(word_map, MAPPING_DUMP_DIR + "/" + name + "_word-map.json")
            
            response = merge_dict(response, invert_dict(word_map, file_name))
        
        result_dump_path = MAPPER_RESULT_DIR + "/" + str(rank) + "_result.json"
        dump_dict(response, result_dump_path)

        comm.send(result_dump_path, dest=MASTER_NODE, tag=COMM_TAG)

    elif rank in REDUCER_NODES:
        files = comm.recv(source=MPI.ANY_SOURCE, tag=COMM_TAG)
        
        working_dict = {}
        for file_path in files:
            working_dict = merge_dict(working_dict, load_dict(file_path))
        
        if rank == REDUCER_NODES[0]:
            dump_dict(working_dict, REDUCER_DATA_DIR + "/reducer_dataset.json")
    
        to_do_letters = get_reducer_letters(rank, REDUCER_NODES)
        log_msg = str(rank) + " -> " + str(to_do_letters) + "\n"
        dump_string(log_msg, REDUCER_DATA_DIR + "/reducers_letters.log")

        results = {}
        for word, files in working_dict.items():
            if any(word.startswith(letter) for letter in to_do_letters):
                result_files = dict(sorted(files.items(), key=lambda item: item[1], reverse=True))
                results[word] = result_files
        
        dump_dict(results, RESULT_DUMP_DIR + "/results_" + str(rank) + ".json")

        comm.send(True, MASTER_NODE, tag=COMM_TAG)
#!/bin/bash

PROC_NUMBER=7
mpiexec --oversubscribe -n $PROC_NUMBER python3 main.py input_dir output_dir
#!/bin/bash

PROC_NUMBER=9
mpiexec --oversubscribe -n $PROC_NUMBER python3 main.py
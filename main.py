from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    print("Hello World from process", rank)
else:
    print("Greeting from process", rank)
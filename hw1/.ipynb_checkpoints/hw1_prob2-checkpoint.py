from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = Get_size()

if rank == 0:
    data_per_rank = 100
    # ints from 1 to 100
    matrix = np.random.rand(1, size * data_per_rank, data_per_rank * size)
    vector = np.random.rand(1, size * data_per_rank, data_per_rank * 1)
else:
    data_per_rank = None

data_per_rank = comm.bcast(data_per_rank, root=0)

if rank != 0:
    vector = np.empty((data_per_rank, 1), dtype=np.int_)

comm.Bcast(vector, root=0)
comm.Bcast(vector, root=0)


from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
# size = Get_size()

if rank == 0:
    dim = 100
    matrix = np.random.randint(0, dim, (dim, dim))
    vector = np.random.randint(0, dim, (dim, 1))
else:
    dim = None

dim = comm.bcast(dim, root=0)

if rank != 0:
    vector = np.empty((dim, 1), dtype=np.int_)

comm.Bcast(vector, root=0)
comm.Bcast(vector, root=0)


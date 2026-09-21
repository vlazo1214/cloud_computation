from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

matrix = None
if rank == 0:
    matrix = np.random.rand(size, size)
    vector = np.random.rand(size, 1)
else:
    vector = np.empty((size, 1), dtype=np.float64)

comm.Bcast(vector, root=0)

rcv_mat = np.empty((1, size), dtype=np.float64)
comm.Scatter(matrix, rcv_mat, root=0)

rcv_res = rcv_mat.dot(vector)

result_vector = None
if rank == 0:
    result_vector = np.empty((size, 1), dtype=np.float64)

comm.Gather(rcv_res, result_vector, root=0)

if rank == 0:
    print("Result shape:", result_vector.shape)

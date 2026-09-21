from mpi4py import MPI
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Global training parameters
L = 0.0001      # Learning Rate
epochs = 10000  # Number of iterations

if rank == 0:
    data = pd.read_csv('20K_Datapoints.csv')
    X_full = data.iloc[:, 0].values.astype(np.float64)
    Y_full = data.iloc[:, 1].values.astype(np.float64)
    n_total = len(X_full)

    # Initial parameters
    params = np.array([0.0, 0.0, 0.0], dtype=np.float64)  # [a, b, c]

    # Calculate data distribution split across MPI ranks
    counts = np.full(size, n_total // size, dtype=int)
    counts[:n_total % size] += 1
    displacements = np.insert(np.cumsum(counts)[:-1], 0, 0)
else:
    X_full = Y_full = None
    params = np.empty(3, dtype=np.float64)
    n_total = None
    counts = displacements = None

# Broadcast metadata to all processes
n_total = comm.bcast(n_total, root=0)
local_count = np.empty(1, dtype=int)
comm.Scatter(counts, local_count, root=0)

# Allocate memory and scatter data subsets
local_X = np.empty(local_count[0], dtype=np.float64)
local_Y = np.empty(local_count[0], dtype=np.float64)

comm.Scatterv([X_full, counts, displacements, MPI.DOUBLE], local_X, root=0)
comm.Scatterv([Y_full, counts, displacements, MPI.DOUBLE], local_Y, root=0)

# Synchronized Gradient Descent Loop
for i in range(epochs):
    # 1. Main process propagates current model parameters to all processes
    comm.Bcast(params, root=0)
    a, b, c = params

    # 2. Each process calculates local predictions and local derivative sums
    Y_pred = a * local_X**2 + b * local_X + c
    error = local_Y - Y_pred

    local_D_a = np.sum(local_X**2 * error)
    local_D_b = np.sum(local_X * error)
    local_D_c = np.sum(error)

    local_grads = np.array([local_D_a, local_D_b, local_D_c], dtype=np.float64)
    global_grads = np.zeros(3, dtype=np.float64)

    # 3. Reduce (sum) gradients from all processes to rank 0
    comm.Reduce(local_grads, global_grads, op=MPI.SUM, root=0)

    # 4. Main process collects total gradients and updates model
    if rank == 0:
        D_a = (-2.0 / n_total) * global_grads[0]
        D_b = (-2.0 / n_total) * global_grads[1]
        D_c = (-2.0 / n_total) * global_grads[2]

        params[0] -= L * D_a
        params[1] -= L * D_b
        params[2] -= L * D_c

if rank == 0:
    a, b, c = params
    print(f"Final Model Parameters: a={a}, b={b}, c={c}")

    # Plot results
    Y_pred_full = a * X_full**2 + b * X_full + c
    plt.scatter(X_full, Y_full, label="Original Data")
    plt.scatter(X_full, Y_pred_full, color='red', label="Predictions")
    plt.legend()
    plt.show()
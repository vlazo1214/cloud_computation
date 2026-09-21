from mpi4py import MPI
import numpy as np
from sklearn.datasets import load_digits
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# 1. Load and broadcast dataset metadata on root
if rank == 0:
    digits = load_digits()
    data = np.ascontiguousarray(digits.data, dtype=np.float64)
    n_samples, n_features = data.shape
    
    n_samples = (n_samples // size) * size
    data = data[:n_samples]
    
    k = 10
    
    # Initialize random centroids
    np.random.seed(42)
    init_indices = np.random.choice(n_samples, k, replace=False)
    centroids = data[init_indices]
else:
    n_samples = n_features = k = None
    data = None
    centroids = None

n_samples = comm.bcast(n_samples, root=0)
n_features = comm.bcast(n_features, root=0)
k = comm.bcast(k, root=0)

if rank != 0:
    centroids = np.empty((k, n_features))

# Scatter data chunks to all processes
local_n = n_samples // size
local_data = np.empty((local_n, n_features))
comm.Scatter(data, local_data, root=0)

begin = time.time()
max_iters = 100

for iteration in range(max_iters):
    # Broadcast current centroids to all processes
    comm.Bcast(centroids, root=0)
    
    # --- E-STEP: Assign local points to nearest centroids ---
    # Compute Euclidean distances: (local_n, k)
    distances = np.linalg.norm(local_data[:, np.newaxis, :] - centroids, axis=2)
    local_labels = np.argmin(distances, axis=1)
    
    # --- M-STEP: Compute local sums and counts for cluster updates ---
    local_sums = np.zeros((k, n_features))
    local_counts = np.zeros((k, 1))
    
    for i in range(k):
        cluster_points = local_data[local_labels == i]
        if len(cluster_points) > 0:
            local_sums[i] = np.sum(cluster_points, axis=0)
            local_counts[i] = len(cluster_points)
            
    # Reduce global sums and counts across all processes
    global_sums = np.zeros_like(local_sums)
    global_counts = np.zeros_like(local_counts)
    
    comm.Allreduce(local_sums, global_sums, op=MPI.SUM)
    comm.Allreduce(local_counts, global_counts, op=MPI.SUM)
    
    # Update centroids globally
    new_centroids = np.zeros_like(centroids)
    for i in range(k):
        if global_counts[i] > 0:
            new_centroids[i] = global_sums[i] / global_counts[i]
        else:
            new_centroids[i] = centroids[i] # Keep old if empty cluster
            
    # Check for convergence
    if np.allclose(centroids, new_centroids):
        break
    centroids = new_centroids

if rank == 0:
    print(f"Parallel K-Means finished in {time.time() - begin:.4f} seconds")

    np.random.seed(42)
    seq_init_indices = np.random.choice(n_samples, k, replace=False)
    seq_centroids = data[seq_init_indices]
    
    for _ in range(max_iters):
        # E-Step
        distances = np.linalg.norm(data[:, np.newaxis, :] - seq_centroids, axis=2)
        seq_labels = np.argmin(distances, axis=1)
        
        # M-Step
        new_seq_centroids = np.zeros_like(seq_centroids)
        for i in range(k):
            cluster_points = data[seq_labels == i]
            if len(cluster_points) > 0:
                new_seq_centroids[i] = np.mean(cluster_points, axis=0)
            else:
                new_seq_centroids[i] = seq_centroids[i]
                
        if np.allclose(seq_centroids, new_seq_centroids):
            break
        seq_centroids = new_seq_centroids

    # --- COMPARE RESULTS ---
    # Note: Centroid rows might occasionally be ordered differently due to floating-point 
    # accumulation differences in Allreduce, but the set of centroids should match.
    # Sorting them ensures alignment for comparison:
    sorted_parallel = centroids[np.argsort(centroids[:, 0])]
    sorted_sequential = seq_centroids[np.argsort(seq_centroids[:, 0])]
    
    is_accurate = np.allclose(sorted_parallel, sorted_sequential, atol=1e-5)
    
    print("\n--- Verification ---")
    print(f"Parallel matches Sequential baseline: {is_accurate}")
    
    if not is_accurate:
        print("Note: Small numerical differences can sometimes occur due to floating-point reduction order in MPI.")

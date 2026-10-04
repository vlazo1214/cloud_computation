import time
import numpy as np
from sklearn.datasets import load_digits
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


def kmeans_step(data_subset, centers):
    """Computes local cluster sums and counts for a subset of data."""
    k = len(centers)
    d = data_subset.shape[1]
    local_sums = np.zeros((k, d))
    local_counts = np.zeros(k)
    
    for x in data_subset:
        distances = np.linalg.norm(centers - x, axis=1)
        cluster_idx = np.argmin(distances)
        local_sums[cluster_idx] += x
        local_counts[cluster_idx] += 1
        
    return local_sums, local_counts

def update_centers(centers, global_sums, global_counts, k):
    """M-step: Updates cluster centers based on global sums and counts."""
    new_centers = np.zeros_like(centers)
    for j in range(k):
        if global_counts[j] > 0:
            new_centers[j] = global_sums[j] / global_counts[j]
        else:
            new_centers[j] = centers[j]
    return new_centers


def run_sequential_kmeans(X, k=10, max_iters=50):
    n_samples, n_features = X.shape
    np.random.seed(42)
    centers = X[np.random.choice(n_samples, k, replace=False)]
    
    for _ in range(max_iters):
        # sequential processes the whole dataset in one go
        global_sums, global_counts = kmeans_step(X, centers)
        
        new_centers = update_centers(centers, global_sums, global_counts, k)
        
        if np.allclose(centers, new_centers):
            break
        centers = new_centers
        
    return centers

def run_threading_kmeans(X, k=10, max_iters=50, num_workers=4):
    n_samples, n_features = X.shape
    np.random.seed(42)
    centers = X[np.random.choice(n_samples, k, replace=False)]
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for _ in range(max_iters):
            chunks = np.array_split(X, num_workers)
            
            # Submit tasks to thread pool
            futures = [executor.submit(kmeans_step, chunk, centers) for chunk in chunks]
            results = [f.result() for f in futures]
            
            # Combine results
            global_sums = np.sum([res[0] for res in results], axis=0)
            global_counts = np.sum([res[1] for res in results], axis=0)
            
            new_centers = update_centers(centers, global_sums, global_counts, k)
            
            if np.allclose(centers, new_centers):
                break
            centers = new_centers
            
    return centers

def run_multiprocessing_kmeans(X, k=10, max_iters=50, num_workers=4):
    n_samples, n_features = X.shape
    np.random.seed(42)
    centers = X[np.random.choice(n_samples, k, replace=False)]
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for _ in range(max_iters):
            chunks = np.array_split(X, num_workers)
            
            # Submit tasks to process pool
            futures = [executor.submit(kmeans_step, chunk, centers) for chunk in chunks]
            results = [f.result() for f in futures]
            
            # Combine results
            global_sums = np.sum([res[0] for res in results], axis=0)
            global_counts = np.sum([res[1] for res in results], axis=0)
            
            new_centers = update_centers(centers, global_sums, global_counts, k)
            
            if np.allclose(centers, new_centers):
                break
            centers = new_centers
            
    return centers
    

if __name__ == "__main__":
    digits = load_digits()
    X = digits.data
    k = 10
    worker_counts = [1, 2, 4, 8]
    
    print("--- Benchmark Results ---")
    
    start = time.time()
    run_sequential_kmeans(X, k=k)
    print(f"Sequential Execution Time: {time.time() - start:.4f} seconds\n")
    
    print("Threading Versions:")
    for w in worker_counts:
        start = time.time()
        run_threading_kmeans(X, k=k, num_workers=w)
        print(f"  Workers: {w} | Time: {time.time() - start:.4f} seconds")

    # Run baseline (1 worker or sequential)
    base_centers = run_threading_kmeans(X, k=10, num_workers=1)
    
    # Check against other worker counts
    for w in [2, 4, 8]:
        worker_centers = run_threading_kmeans(X, k=10, num_workers=w)
        
        # Are the final centers effectively identical?
        are_results_identical = np.allclose(base_centers, worker_centers, atol=1e-5)
        
        print(f"Workers: {w} | Results match 1-worker baseline? {are_results_identical}")
    
    print("\nMultiprocessing Versions:")
    for w in worker_counts:
        start = time.time()
        run_multiprocessing_kmeans(X, k=k, num_workers=w)
        print(f"  Workers: {w} | Time: {time.time() - start:.4f} seconds")

    
    base_centers = run_multiprocessing_kmeans(X, k=10, num_workers=1)
    
    for w in [2, 4, 8]:
        worker_centers = run_multiprocessing_kmeans(X, k=10, num_workers=w)
        
        are_results_identical = np.allclose(base_centers, worker_centers, atol=1e-5)
        
        print(f"Workers: {w} | Results match 1-worker baseline? {are_results_identical}")
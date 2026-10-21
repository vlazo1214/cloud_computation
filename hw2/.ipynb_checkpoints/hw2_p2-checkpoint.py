import time
import sys
import multiprocessing as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def worker_shared_fast(rank, num_processes, X, Y, shared_params, shared_grads, epochs, barrier):
    chunk_size = len(X) // num_processes
    start_idx = rank * chunk_size
    end_idx = len(X) if rank == num_processes - 1 else (rank + 1) * chunk_size
    
    X_local = X[start_idx:end_idx]
    Y_local = Y[start_idx:end_idx]
    
    # Pre-calculate offsets for this worker's local gradient slot
    offset = rank * 3
    
    for _ in range(epochs):
        # 1. Read global parameters (Lock-free read)
        curr_a = shared_params[0]
        curr_b = shared_params[1]
        curr_c = shared_params[2]
        
        # 2. Compute local gradients
        Y_pred = curr_a * X_local * X_local + curr_b * X_local + curr_c
        
        # 3. Write directly into dedicated non-overlapping slots (No Lock required!)
        shared_grads[offset]     = np.sum(X_local * X_local * (Y_local - Y_pred))
        shared_grads[offset + 1] = np.sum(X_local * (Y_local - Y_pred))
        shared_grads[offset + 2] = np.sum(Y_local - Y_pred)
        
        # 4. Wait for all workers to finish gradient calculation
        barrier.wait()
        
        # 5. Worker 0 aggregates gradients and updates shared parameters
        if rank == 0:
            total_da = sum(shared_grads[i * 3]     for i in range(num_processes))
            total_db = sum(shared_grads[i * 3 + 1] for i in range(num_processes))
            total_dc = sum(shared_grads[i * 3 + 2] for i in range(num_processes))
            
            # Update shared parameters
            shared_params[0] -= 0.0001 * (-2 / len(X)) * total_da
            shared_params[1] -= 0.0001 * (-2 / len(X)) * total_db
            shared_params[2] -= 0.0001 * (-2 / len(X)) * total_dc
            
        # 6. Wait for Worker 0 to finish updating before starting the next epoch
        barrier.wait()


def run_shared_memory(X, Y, a, b, c, L, epochs, n, num_processes=4):
    start_time = time.time()
    
    # Shared memory arrays:
    # shared_params: [a, b, c]
    # shared_grads: [d_a_0, d_b_0, d_c_0, d_a_1, d_b_1, d_c_1, ...]
    shared_params = mp.Array('d', [a, b, c])
    shared_grads = mp.Array('d', [0.0] * (3 * num_processes))
    
    # Single barrier with total count = num_processes
    barrier = mp.Barrier(num_processes)
    
    processes = []
    for rank in range(num_processes):
        p = mp.Process(
            target=worker_shared_fast, 
            args=(rank, num_processes, X, Y, shared_params, shared_grads, epochs, barrier)
        )
        processes.append(p)
        p.start()
        
    for p in processes:
        p.join()
        
    end_time = time.time()
    print(f"[Shared Memory] Runtime: {end_time - start_time:.4f} seconds")
    return shared_params[0], shared_params[1], shared_params[2]

X_global, Y_global = None, None

def init_pool(X_data, Y_data):
    global X_global, Y_global
    X_global, Y_global = X_data, Y_data

def compute_local_gradient_pool(args):
    rank, num_processes, a, b, c = args
    chunk_size = len(X_global) // num_processes
    start_idx = rank * chunk_size
    end_idx = len(X_global) if rank == num_processes - 1 else (rank + 1) * chunk_size
    
    X_loc = X_global[start_idx:end_idx]
    Y_loc = Y_global[start_idx:end_idx]
    
    Y_pred = a * X_loc * X_loc + b * X_loc + c
    d_a = np.sum(X_loc * X_loc * (Y_loc - Y_pred))
    d_b = np.sum(X_loc * (Y_loc - Y_pred))
    d_c = np.sum(Y_loc - Y_pred)
    
    return d_a, d_b, d_c

def run_pool(X, Y, a, b, c, L, epochs, n, num_processes=4):
    start_time = time.time()
    
    with mp.Pool(processes=num_processes, initializer=init_pool, initargs=(X, Y)) as pool:
        for _ in range(epochs):
            tasks = [(i, num_processes, a, b, c) for i in range(num_processes)]
            results = pool.map(compute_local_gradient_pool, tasks)
            
            total_da = sum(r[0] for r in results)
            total_db = sum(r[1] for r in results)
            total_dc = sum(r[2] for r in results)
            
            a -= L * (-2 / n) * total_da
            b -= L * (-2 / n) * total_db
            c -= L * (-2 / n) * total_dc
            
    end_time = time.time()
    print(f"[Pool] Runtime: {end_time - start_time:.4f} seconds")
    return a, b, c


def worker_queue(rank, num_processes, X, Y, task_queue, result_queue, epochs):
    chunk_size = len(X) // num_processes
    start_idx = rank * chunk_size
    end_idx = len(X) if rank == num_processes - 1 else (rank + 1) * chunk_size
    X_loc = X[start_idx:end_idx]
    Y_loc = Y[start_idx:end_idx]
    
    for _ in range(epochs):
        params = task_queue.get()
        if params is None:
            break
        a, b, c = params
        Y_pred = a * X_loc * X_loc + b * X_loc + c
        d_a = np.sum(X_loc * X_loc * (Y_loc - Y_pred))
        d_b = np.sum(X_loc * (Y_loc - Y_pred))
        d_c = np.sum(Y_loc - Y_pred)
        result_queue.put((d_a, d_b, d_c))

def run_queue(X, Y, a, b, c, L, epochs, n, num_processes=4):
    start_time = time.time()
    
    task_queues = [mp.Queue() for _ in range(num_processes)]
    result_queue = mp.Queue()
    
    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=worker_queue, args=(rank, num_processes, X, Y, task_queues[rank], result_queue, epochs))
        processes.append(p)
        p.start()
        
    for _ in range(epochs):
        for q in task_queues:
            q.put((a, b, c))
            
        total_da, total_db, total_dc = 0.0, 0.0, 0.0
        for _ in range(num_processes):
            da, db, dc = result_queue.get()
            total_da += da
            total_db += db
            total_dc += dc
            
        a -= L * (-2 / n) * total_da
        b -= L * (-2 / n) * total_db
        c -= L * (-2 / n) * total_dc
        
    for q in task_queues:
        q.put(None)
    for p in processes:
        p.join()
        
    end_time = time.time()
    print(f"[Queue] Runtime: {end_time - start_time:.4f} seconds")
    return a, b, c


def worker_pipe(rank, num_processes, X, Y, conn, epochs):
    chunk_size = len(X) // num_processes
    start_idx = rank * chunk_size
    end_idx = len(X) if rank == num_processes - 1 else (rank + 1) * chunk_size
    X_loc = X[start_idx:end_idx]
    Y_loc = Y[start_idx:end_idx]
    
    for _ in range(epochs):
        params = conn.recv()
        if params is None:
            break
        a, b, c = params
        Y_pred = a * X_loc * X_loc + b * X_loc + c
        d_a = np.sum(X_loc * X_loc * (Y_loc - Y_pred))
        d_b = np.sum(X_loc * (Y_loc - Y_pred))
        d_c = np.sum(Y_loc - Y_pred)
        conn.send((d_a, d_b, d_c))

def run_pipe(X, Y, a, b, c, L, epochs, n, num_processes=4):
    start_time = time.time()
    
    parent_conns, worker_conns = zip(*[mp.Pipe() for _ in range(num_processes)])
    
    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=worker_pipe, args=(rank, num_processes, X, Y, worker_conns[rank], epochs))
        processes.append(p)
        p.start()
        
    for _ in range(epochs):
        for conn in parent_conns:
            conn.send((a, b, c))
            
        total_da, total_db, total_dc = 0.0, 0.0, 0.0
        for conn in parent_conns:
            da, db, dc = conn.recv()
            total_da += da
            total_db += db
            total_dc += dc
            
        a -= L * (-2 / n) * total_da
        b -= L * (-2 / n) * total_db
        c -= L * (-2 / n) * total_dc
        
    for conn in parent_conns:
        conn.send(None)
    for p in processes:
        p.join()
        
    end_time = time.time()
    print(f"[Pipe] Runtime: {end_time - start_time:.4f} seconds")
    return a, b, c


if __name__ == '__main__':
    # Load dataset
    data = pd.read_csv('20K_Datapoints.csv')
    X = data.iloc[:, 0].values
    Y = data.iloc[:, 1].values

    # Hyperparameters
    L = 0.0001
    epochs = 1000  # Reduced epochs for faster comparison; set back to 10000 if needed
    n = float(len(X))
    num_processes = 4

    # Choose which method to run ('sequential', 'shared_memory', 'pool', 'queue', 'pipe')
    if len(sys.argv) != 2:
        print("Add one of these methods as a command line argument: \'sequential\', \'shared_memory\', \'pool\', \'queue\', \'pipe\'")
        exit()

    method = sys.argv[1]
    
    # Initial weights
    a, b, c = 0.0, 0.0, 0.0

    print(f"Running method: {method.upper()} with {num_processes} processes...")
    
    if method == 'sequential':
        a, b, c = run_sequential(X, Y, a, b, c, L, epochs, n)
    elif method == 'shared_memory':
        a, b, c = run_shared_memory(X, Y, a, b, c, L, epochs, n, num_processes)
    elif method == 'pool':
        a, b, c = run_pool(X, Y, a, b, c, L, epochs, n, num_processes)
    elif method == 'queue':
        a, b, c = run_queue(X, Y, a, b, c, L, epochs, n, num_processes)
    elif method == 'pipe':
        a, b, c = run_pipe(X, Y, a, b, c, L, epochs, n, num_processes)
    else:
        raise ValueError("Invalid method specified.")

    print(f"Final Model Parameters: a={a}, b={b}, c={c}")

    # Plot results
    Y_pred = a * X * X + b * X + c
    plt.scatter(X, Y, alpha=0.5, label='Actual')
    plt.scatter(X, Y_pred, color='red', s=1, label='Predicted')
    plt.legend()
    plt.show()
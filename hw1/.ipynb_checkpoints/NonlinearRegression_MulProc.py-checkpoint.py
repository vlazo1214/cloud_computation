import multiprocessing as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def worker_process(pipe, local_X, local_Y):
    """Worker task that runs on each CPU core."""
    while True:
        message = pipe.recv()
        if message == "STOP":
            break
        
        # Unpack parameters broadcasted from the main process
        a, b, c = message
        
        # Calculate local error and derivative sums
        Y_pred = a * local_X**2 + b * local_X + c
        error = local_Y - Y_pred

        local_D_a = np.sum(local_X**2 * error)
        local_D_b = np.sum(local_X * error)
        local_D_c = np.sum(error)

        # Send local gradient sums back to the main process
        pipe.send((local_D_a, local_D_b, local_D_c))

if __name__ == '__main__':
    # Preprocessing Input data
    data = pd.read_csv('20K_Datapoints.csv')
    X_full = data.iloc[:, 0].values.astype(np.float64)
    Y_full = data.iloc[:, 1].values.astype(np.float64)
    n_total = float(len(X_full))

    # Gradient descent parameters
    a, b, c = 0.0, 0.0, 0.0
    L = 0.0001
    epochs = 10000

    num_workers = mp.cpu_count()
    
    # Partition data among worker processes
    X_splits = np.array_split(X_full, num_workers)
    Y_splits = np.array_split(Y_full, num_workers)

    pipes = []
    processes = []

    # Spawn worker processes
    for i in range(num_workers):
        parent_conn, child_conn = mp.Pipe()
        p = mp.Process(target=worker_process, args=(child_conn, X_splits[i], Y_splits[i]))
        pipes.append(parent_conn)
        processes.append(p)
        p.start()

    # Synchronized Gradient Descent Loop
    for epoch in range(epochs):
        # 1. Send current model parameters to all processes
        params = (a, b, c)
        for pipe in pipes:
            pipe.send(params)

        # 2. Main process collects gradients from all worker processes
        total_D_a, total_D_b, total_D_c = 0.0, 0.0, 0.0
        for pipe in pipes:
            local_D_a, local_D_b, local_D_c = pipe.recv()
            total_D_a += local_D_a
            total_D_b += local_D_b
            total_D_c += local_D_c

        # 3. Main process summarizes gradients and updates the model
        D_a = (-2 / n_total) * total_D_a
        D_b = (-2 / n_total) * total_D_b
        D_c = (-2 / n_total) * total_D_c

        a -= L * D_a
        b -= L * D_b
        c -= L * D_c

    # Clean up worker processes
    for pipe in pipes:
        pipe.send("STOP")
    for p in processes:
        p.join()

    print(f"Final Model Parameters: a={a}, b={b}, c={c}")

    # Plot results
    Y_pred = a * X_full**2 + b * X_full + c
    plt.scatter(X_full, Y_full, label="Original Data")
    plt.scatter(X_full, Y_pred, color='red', label="Predictions")
    plt.legend()
    plt.show()
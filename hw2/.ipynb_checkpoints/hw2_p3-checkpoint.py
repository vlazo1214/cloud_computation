import time
import numpy as np
from numba.openmp import njit, openmp_context as openmp

# 1. Must be decorated with @njit from numba.openmp
@njit
def compute_gradients_parallel(x, y, a, b, c):
    N = len(x)
    da_total = 0.0
    db_total = 0.0
    dc_total = 0.0

    # 2. Use the openmp context manager with standard directives
    with openmp("parallel for reduction(+:da_total,db_total,dc_total) schedule(static)"):
        for i in range(N):
            pred = a * np.exp(-b * x[i]) + c
            error = pred - y[i]
            
            grad_a = 2.0 * error * np.exp(-b * x[i])
            grad_b = 2.0 * error * (-a * x[i] * np.exp(-b * x[i]))
            grad_c = 2.0 * error * 1.0

            da_total += grad_a
            db_total += grad_b
            dc_total += grad_c

    return da_total / N, db_total / N, dc_total / N

def train_parallel(x, y, epochs=500, lr=0.01, num_threads=4):
    a, b, c = 1.0, 1.0, 1.0
    
    # Set OpenMP thread count via standard environment variable or runtime
    import os
    os.environ["OMP_NUM_THREADS"] = str(num_threads)

    start_time = time.perf_counter()

    for epoch in range(epochs):
        da, db, dc = compute_gradients_parallel(x, y, a, b, c)
        a -= lr * da
        b -= lr * db
        c -= lr * dc

    runtime = time.perf_counter() - start_time
    return (a, b, c), runtime
    
if __name__ == "__main__":
    # Generate synthetic training data
    np.random.seed(42)
    x = np.linspace(0, 5, 100000)
    y = 10.0 * np.exp(-6.0 * x) + 2.8 + np.random.normal(0, 0.1, size=len(x))

    thread_counts = [1, 2, 4, 8]
    runtimes = {}
    speedups = {}

    print(f"{'Threads':<10}{'Runtime (s)':<15}{'Speedup':<12}{'Final Parameters (a, b, c)'}")
    print("-" * 65)

    base_time = None

    for threads in thread_counts:
        params, rtime = train_parallel(x, y, epochs=500, lr=0.01, num_threads=threads)
        runtimes[threads] = rtime

        if threads == 1:
            base_time = rtime
            speedups[threads] = 1.0
        else:
            speedups[threads] = base_time / rtime

        print(f"{threads:<10}{rtime:<15.4f}{speedups[threads]:<12.2f}a={params[0]:.4f}, b={params[1]:.4f}, c={params[2]:.4f}")
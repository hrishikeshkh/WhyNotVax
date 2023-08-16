import os
import multiprocessing

def get_num_cores():
    num_cores = multiprocessing.cpu_count()
    return num_cores

if __name__ == "__main__":
    num_cores = get_num_cores()
    print(f"Your computer has {num_cores} CPU cores.")

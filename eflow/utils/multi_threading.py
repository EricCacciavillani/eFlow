import multiprocessing as mp

def multiprocess(func,
                 jobs,
                 cores):
    results = []

    if cores == 1:
        for j in jobs:
            results.append(func(j))
    elif cores == -1:
        with mp.Pool(mp.cpu_count()) as p:
            results = list(p.map(func, jobs))
    elif cores > 1:
        with mp.Pool(cores) as p:
            results = list(p.map(func, jobs))
    else:
        print('Error: cores must be a integer')

    return results
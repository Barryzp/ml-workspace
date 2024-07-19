import numpy as np
from ppso import ppso

def main():
    np.random.seed(74)
    rt = 10
    layers = [4, 8, 20, 32]
    ps = sum(layers)
    d = 30
    MAX_FES = d * 10000
    funcid = 1
    lb = -600
    ub = 600
    results = np.ones(rt) * 99999999999

    # 这代表独立重复多少次实验
    for ri in range(rt):
        results[ri], fitness, pop, fbest = ppso(layers, d, lb, ub, MAX_FES, funcid)
        print(f'{ri + 1} : {results[ri]:e}')
    
    print('\n\n====================\n\n')
    print(f'FID:{funcid} mean result: {np.mean(results):e}')
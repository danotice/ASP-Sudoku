import numpy as np
import pandas as pd
import multiprocessing as mp

# Setting up puzzles
#input to array
def format_Puzzle(array,N=9):
    output = array.replace('.','0')
    output = np.array(list(output), dtype=int)
    output = np.reshape(output, (N,N))

    return output

def print_Puzzle(puzzle):
    N = len(puzzle)
    n = int(np.sqrt(N))
    breaks = [i-1 for i in range(n,N,n)]
    
    for r in range(N):
        for c in range(N):
            print(str(puzzle[r,c]) + ' ', end='', flush=True) 
            if c in breaks:
                print('| ', end='')
        print()
        if r in breaks:
            print('-------------------')              
    
def get_Subgrids(n):
    #list of all subgrid indices
    subgrids = list()
    for I in range(0,n*n,n):
        for J in range(0,n*n,n):
            subgrids.append([(i,j) for i in range(I,I+n) for j in range(J,J+n)])
    return subgrids


def GetNonFixedCells(empty, N, bygrid = True):
    n = int(np.sqrt(N))

    if bygrid:
        fixed = list()
        for I in range(0,N,n):
            for J in range(0,N,n):
                fixed.append([(i,j) for i in range(I,I+n) for j in range(J,J+n) if empty[i,j]==0])
    else:
        fixed = [(i,j) for i in range(N) for j in range(N) if empty[i,j]==0]
    
    
    return fixed

def GetFixedCells(empty, N, bygrid = True):
    n = int(np.sqrt(N))

    if bygrid:
        fixed = list()
        for I in range(0,N,n):
            for J in range(0,N,n):
                fixed.append([(i,j) for i in range(I,I+n) for j in range(J,J+n) if empty[i,j]!=0])
    else:
        fixed = [(i,j) for i in range(N) for j in range(N) if empty[i,j]!=0] 
    
    return fixed

def GetNonFixedCellsFlat(empty):
    puzzleFlat = empty.flat
    return [i for (i,c) in enumerate(puzzleFlat) if c==0]

def GetNonFixedIndSG(empty, N):
    return [[j for j in range(N) if empty[i,j]==0] for i in range(N) ]

def parallelize_dataframe(df, rng_streams, func, n_cores=4):    

    df_split = np.array_split(df, n_cores)
    pool = mp.Pool(n_cores)
    df = pd.concat(pool.starmap(func, list(zip(df_split,rng_streams))))
    pool.close()
    pool.join()
    return df

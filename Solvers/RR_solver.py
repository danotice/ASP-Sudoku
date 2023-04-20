import numpy as np
from numpy.random import SeedSequence, default_rng
from copy import deepcopy
import pandas as pd
import multiprocessing as mp
import sudoku as s


#things to change
inpath = './Data/puzzles1000.pkl'
outfile = "./Data/Performance Data/RR_perf.pkl"
maxits = 500000
Runs = 20
ncores = 25

subgrids = s.get_Subgrids(3)

subgrid_dict = {}
for i in range(9):
    for j in subgrids[i]:
        subgrid_dict[j] = i


def CostFull(soln):
    N = len(soln)
    A = set(range(1,N+1))
    # looks at each row individually and calculates the number of values from 1-9 missing. same for column
    cost = 0
    for i in range(N):
        cost += len(A - set(soln[i])) + len(A - set(np.transpose(soln)[i])) # rows and columns
        cost += len(A - {soln[r,c] for (r,c) in subgrids[i]}) #subgrids
    
    return cost

def CostUpdate(oldSol, newSol, changedCells):
    # for Swaps only
    N = len(oldSol)
    A = set(range(1,N+1))
    cells = [(i,j) for i in range(N) for j in range(N)]

    before = 0
    after = 0

    rows = {cells[i][0] for i in changedCells}
    cols = {cells[i][1] for i in changedCells}
    sub = {subgrid_dict[cells[i]] for i in changedCells}
    #print('r', rows, 'c', cols, 's', sub)

    for r in rows:
        before += len(A - set(oldSol[r]))
        after += len(A - set(newSol[r]))
        #print(r, before, after)

    for c in cols:
        before += len(A - set(np.transpose(oldSol)[c]))
        after += len(A - set(np.transpose(newSol)[c]))
        #print(c, before, after)

    for s in sub:
        before += len(A - {oldSol[i,j] for (i,j) in subgrids[s]})
        after += len(A - {newSol[i,j] for (i,j) in subgrids[s]})
        #print(s, before, after)

    
    return after - before
    
def parallelize_dataframe(df, rng_streams, func, n_cores):    

    df_split = np.array_split(df, n_cores)
    pool = mp.Pool(n_cores)
    df = pd.concat(pool.starmap(func, list(zip(df_split,rng_streams))))
    pool.close()
    pool.join()
    return df

def RR_cols(df, rng, dev=1, lim=maxits,runs=Runs):
    rng = rng

        # RR solver - maxits, takes initial solution, update cost
    def RR_solverIts(empty, init, dev, prob, maxit):
        
        #N = len(empty)
        current = deepcopy(init)
        poolf  = s.GetNonFixedCellsFlat(empty)

        f_current = CostFull(current)
        its = 0
        k_best = 0
        #Fs = [f_current]

        f_best = f_current
        best_s = deepcopy(current)
        prev = deepcopy(current)

        
        while not f_current == 0 and (its < maxit):
                
            a,b = rng.choice(poolf,2,replace=False)
            current.flat[a], current.flat[b] =  current.flat[b], current.flat[a]
            #f_new = CostFull(current)
            f_new = f_current + CostUpdate(prev, current, [a,b])
            
            if rng.random() < prob:
                continue

            its += 1 
            if f_new <= f_current or f_new <= f_best + dev:
                f_current = f_new
                prev = deepcopy(current)

                if f_new < f_best:
                    f_best = f_new
                    best_s = deepcopy(current) 
                    k_best = its               
        
            else: #to go back to previous solution
                current = deepcopy(prev)
            #Fs.append(f_current)
            
        #print(its)
        return best_s, f_best, k_best, its              



    # multirun, multi-time - can only take solvers with initial solutions
    def multiRun_solverM(puzzle, init, solver, runs, dev, prob, cond):
        # check that any optimal soln are the same
        sol = [[]]
        
        total_its = []
        best_its = []
        obj = []

        for i in range(runs):    
            #print(i)
            s,fbest,kbest, k = solver(puzzle, init[i], dev, prob, cond)

            if fbest==0 and not np.array_equal(sol[-1],s):
                sol.append(s)
            total_its.append(k)
            best_its.append(kbest)
            obj.append(fbest)

        return sol, total_its, best_its, obj


    df[['RR_sols','RR_its','RR_bestIts','RR_obj']] = df.apply(
        lambda row: multiRun_solverM(row.puzzlesF, row.solnInit, RR_solverIts, runs, dev, 0,lim),
        axis=1, result_type='expand')
    return df


if __name__ == "__main__":

    samplePuzzles = pd.read_pickle(inpath)
    #samplePuzzles = samplePuzzles.iloc[:4,:]


    child_seeds = SeedSequence(1111).spawn(ncores)
    rng_streams1 = [default_rng(s) for s in child_seeds]

    Results = parallelize_dataframe(samplePuzzles, rng_streams1, RR_cols, ncores)
    pd.to_pickle(Results, outfile)

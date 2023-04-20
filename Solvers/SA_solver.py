import numpy as np
from numpy.random import SeedSequence, default_rng
from copy import deepcopy
import pandas as pd
import sudoku as s

#### things to change
inpath = './Data/puzzles1000.pkl'
outfile = "./Data/Performance Data/SA_perf.pkl"
maxit = 500000
Runs = 20
ncores = 25
######

subgrids = s.get_Subgrids(3)

subgrid_dict = {}
for i in range(9):
    for j in subgrids[i]:
        subgrid_dict[j] = i


def CostFull(soln, N):
    #N = len(soln)
    A = set(range(1,N+1))
    # looks at each row individually and calculates the number of values from 1-9 missing. same for column
    cost = 0
    for i in range(N):
        cost += len(A - set(soln[i])) + len(A - set(np.transpose(soln)[i])) # rows and columns
        cost += len(A - {soln[r,c] for (r,c) in subgrids[i]}) #subgrids
    
    return cost


def CostUpdate(oldSol, newSol, changedCells, N):
    # for Global Swaps only
    #N = len(oldSol)
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


def Solver_cols(df, rng, alpha=0.9, rh=20, lim=maxit,runs=Runs):
    rng = rng

    def GenerateInitialSolution(empty, N):
        initialSol = deepcopy(empty)
        
        for n in range(N):
            missing = list(set(range(1,N+1)) - {empty[i,j] for (i,j) in subgrids[n]})
            #np.random.shuffle(missing) #this is shuffling inplace
            rng.shuffle(missing)

            for (i,j) in subgrids[n]:
                if initialSol[i,j] == 0:
                    initialSol[i,j] = missing.pop(0)

        return initialSol


    def Swap(soln, a,b):
        neighbour = deepcopy(soln)    
        nflat = neighbour.flat
        nflat[a], nflat[b] = nflat[b], nflat[a]        

        return neighbour

    def SetInitialTemperature(initSol, N, poolf):
        
        sol = initSol
        swaps = [rng.choice(poolf,2,replace=False) for _ in range(50)]
        Costs = [CostFull(Swap(sol, a,b),N) for (a,b) in swaps]
        
        return np.std(Costs)
        
    def AdaptTemperature(T, alpha):
        return alpha*T

    
    # SA with global swaps - its, updating costs
    def SAglob_Solver_its(empty,initSol, alpha, rh, maxit): 
        N = len(empty)
        poolf  = s.GetNonFixedCellsFlat(empty)
        nonFixed = s.GetNonFixedCells(empty, N, True)

        ml = np.square(np.sum([len(box) for box in nonFixed]))
        chains = 0
        
        current = initSol
        f_current = CostFull(current, N)
        f_best = f_current
        best_s = deepcopy(current)
        k_best = 0

        k = 0
        T0 = SetInitialTemperature(initSol, N, poolf)
        T = T0
        #Fs = [f_current]
        
        c=0

        while not f_current == 0 and (k < maxit):
            
            a,b = rng.choice(poolf,2,replace=False)    
            candidate = Swap(current, a,b)
            f_new = f_current + CostUpdate(current,candidate,[a,b],N)
            
            if f_new < f_current:
                f_current = f_new
                current = deepcopy(candidate)
                chains = 0
                if f_new < f_best:
                    f_best = f_new
                    best_s = deepcopy(current)
                    k_best = k      

            elif np.exp((f_current - f_new)/T) > rng.random():
                f_current = f_new
                current = deepcopy(candidate)
                    
            c += 1

            if c > ml: #to have several iterations at same temperature
                chains += 1
                c = 0
                #print(f'end chain {chains}')

                if chains == rh:  
                    #print('reheat')              
                    T = T0
                    current = GenerateInitialSolution(empty,N) 
                    f_current = CostFull(current,N)      
                    chains=0         
                else:
                    T = AdaptTemperature(T,alpha)            
                
            k += 1
        
            #Fs.append(f_current)
            
        return best_s, f_best, k_best, k 
        
    # multirun, multi-time - can only take solvers with initial solutions
    def multiRun_solverM(puzzle, init, solver, runs, alpha, rh, cond):
        # check that any optimal soln are the same
        sol = [[]]
        
        total_its = []
        best_its = []
        obj = []

        for i in range(runs):    
            #print(i)
            s,fbest,kbest, k = solver(puzzle, init[i], alpha, rh, cond)

            if fbest==0 and not np.array_equal(sol[-1],s):
                sol.append(s)
            total_its.append(k)
            best_its.append(kbest)
            obj.append(fbest)

        return sol, total_its, best_its, obj
                    

    df[['SAglob_sols','SAglob_its','SAglob_bestIts','SAglob_obj']] = df.apply(
        lambda row: multiRun_solverM(row.puzzlesF, row.solnInit, SAglob_Solver_its, runs, alpha, rh, lim),
        axis=1, result_type='expand')
    return df


if __name__ == "__main__":

    samplePuzzles = pd.read_pickle(inpath)
    #samplePuzzles = samplePuzzles.iloc[:4,:]


    child_seeds = SeedSequence(1111).spawn(ncores)
    rng_streams1 = [default_rng(s) for s in child_seeds]

    Results = s.parallelize_dataframe(samplePuzzles, rng_streams1, Solver_cols, ncores)
    pd.to_pickle(Results, outfile)

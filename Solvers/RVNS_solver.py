import numpy as np
from numpy.random import SeedSequence, default_rng
from copy import deepcopy
import pandas as pd
import sudoku as s

#### things to change
inpath = './Data/puzzles1000.pkl'
outfile = "./Data/Performance Data/RVNS_perf.pkl"
maxit = 500000
Runs = 20
ncores = 25
######

subgrids = s.get_Subgrids(3) #probably need to change so that subgrids is defined inside


#switching views
def to_subgrid_view(puzzle):
    N = len(puzzle)
    n = int(np.sqrt(N))

    reshaped = [puzzle[x+j*n, i*n:i*n + n] for j in range(n) for i in range(n) for x in range(n)] 
    reshaped = [[x for xs in reshaped[i*n:i*n + n] for x in xs] for i in range(N)]
    
    return np.array(reshaped)

def to_row_view(puzzle):
    N = len(puzzle)
    reshaped = np.zeros((N,N),int)
    for G in range(N):
        for c in range(N):
            reshaped[subgrids[G][c]] = puzzle[G,c]

    return reshaped

def CostFullSG(solSG,N): #rename to Cost
    #takes soln in subgrid view
    soln = to_row_view(solSG)
    #N = len(soln)
    A = set(range(1,N+1))
    # looks at each row individually and calculates the number of values from 1-9 missing. same for column
    cost = 0
    for i in range(N):
        cost += len(A - set(soln[i])) + len(A - set(np.transpose(soln)[i])) # rows and columns
        cost += len(A - {soln[r,c] for (r,c) in subgrids[i]}) #subgrids
    
    return cost

def Solver_cols(df, rng, lim=maxit, runs=Runs):
    rng = rng
        
    def Swap(soln, sub, a,b):
        #soln input and output in subgrid view
        neighbour = deepcopy(soln)    
        neighbour[sub,a], neighbour[sub,b] = neighbour[sub,b], neighbour[sub,a]        
        return neighbour


    def SwapRand(soln, nonFixed,N):
        box = rng.integers(N)

        if len(nonFixed[box])<= 1: #cant swap only 1 fixed cell
            return soln 
        
        a,b = rng.choice(nonFixed[box],size=2,replace=False)
        return Swap(soln, box, a,b)
            

    def Insert(soln, sub, a, b, nonfixed): 
        #inserts value at a just before cell b
        #pool is the list of non-fixed indices in the subgrid
        #both a and b must be non-fixed
        #takes soln in subgrid view
        pool = nonfixed[sub]
            
        neighbour = deepcopy(soln)  
        A,B = soln[sub,[a,b]]  #values of cells
        a_pool = list(soln[sub,pool]).index(A) #position in the list of movable cells
        b_pool = list(soln[sub,pool]).index(B)

        if a_pool < b_pool:
            neighbour[sub,pool] = np.insert(np.delete(soln[sub,pool], a_pool),b_pool-1,A)
        else:
            neighbour[sub,pool] = np.insert(np.delete(soln[sub,pool], a_pool),b_pool,A)

        return neighbour


    def InsertRand(soln, nonfixed,N):
        box = rng.integers(N)

        if len(nonfixed[box])== 0: #cant swap only 1 fixed cell
            return soln 
        
        a,b = rng.choice(nonfixed[box],size=2,replace=False) # must be 2 nonfixed cells
        return Insert(soln,box,a,b,nonfixed)
        
    def CPOEx(soln, c, sub, nonfixed,N):
        neighbour = deepcopy(soln)
        pool = nonfixed[sub]
        
        i = 1
        while c+i < N and c-i >= 0 and c+i in pool and c-i in pool:
            neighbour[sub,c-i], neighbour[sub,c+i] = neighbour[sub,c+i], neighbour[sub,c-i]
            i += 1

        return neighbour

    def CPOExRand(soln, nonfixed,N):
        box = rng.integers(N)

        if len(nonfixed[box])== 0: #cant swap only 1 fixed cell
            return soln 
        
        c = rng.choice(range(2,N-1))
        return CPOEx(soln,c,box,nonfixed,N)

    def Invert(soln, sub,a,b, nonfixed):
        
        neighbour = deepcopy(soln)
        pool = [x for x in nonfixed[sub] if x >= a and x<=b]

        cells = neighbour[sub,pool] 
        neighbour[sub, pool] = cells[::-1]

        return neighbour

    def InvertRand(soln, nonfixed,N):
        box = rng.integers(N)
        
        if len(nonfixed[box])== 0: #cant swap only 1 fixed cell
            return soln 
        
        a,b = sorted(rng.choice(N,2,replace=False))
        return Invert(soln, box, a,b, nonfixed)

    def SwapGlob(soln, a,b):
        neighbour = deepcopy(soln)    
        nflat = neighbour.flat
        nflat[a], nflat[b] = nflat[b], nflat[a]        

        return neighbour

    def SwapGlobRand(soln,poolf):
        a,b = rng.choice(poolf,2,replace=False)    
        return SwapGlob(soln,a,b)

    def RVNS_Solver_its(empty, initSol, maxit):
        N = len(empty)
        k_max = 5

        emptySG = to_subgrid_view(empty)
        initSG = to_subgrid_view(initSol)

        nonfixed = s.GetNonFixedIndSG(emptySG,N)
        poolf  = s.GetNonFixedCellsFlat(emptySG)

        def Neighbour(sol, i):
            """Returns a random point in the i-th neighbourhood of sol"""
            if i==0:
                return InvertRand(sol,nonfixed,N)
            if i==1:
                return CPOExRand(sol,nonfixed,N)
            if i==2:
                return InsertRand(sol,nonfixed,N)
            if i==3:
                return SwapRand(sol,nonfixed,N)    
            if i==4:
                return SwapGlobRand(sol,poolf)
        
        current = initSG
        f_current = CostFullSG(current,N)
        f_best = f_current
        best_s = deepcopy(current)
        its_best = 0
        its = 0

        
        while f_current > 0 and its < maxit:
            k = 0

            while k != k_max:
                candidate = Neighbour(current,k)
                f_new = CostFullSG(candidate,N)
                its += 1

                if f_new <= f_current:
                    current = deepcopy(candidate)
                    f_current = f_new
                    k = 0
                    if f_new < f_best:
                        f_best = f_new
                        best_s = deepcopy(current)
                        its_best = its
                        if f_best == 0:
                            break

                else:
                    k += 1
            
        return to_row_view(best_s), f_best, its_best, its
        
    # multirun, multi-time - can only take solvers with initial solutions
    def multiRun_solverM(puzzle, init, solver, runs, cond):
        # check that any optimal soln are the same
        sol = [[]]
        
        total_its = []
        best_its = []
        obj = []

        for i in range(runs):    
            #print(i)
            s,fbest,kbest, k = solver(puzzle, init[i], cond)

            if fbest==0 and not np.array_equal(sol[-1],s):
                sol.append(s)
            total_its.append(k)
            best_its.append(kbest)
            obj.append(fbest)

        return sol, total_its, best_its, obj
            
    df[['RVNS_sols','RVNS_its','RVNS_bestIts','RVNS_obj']] = df.apply(
        lambda row: multiRun_solverM(row.puzzlesF, row.solnInit, RVNS_Solver_its, runs, lim),
        axis=1, result_type='expand')
    return df


if __name__ == "__main__":

    samplePuzzles = pd.read_pickle(inpath)
    #samplePuzzles = samplePuzzles.iloc[:4,:]


    child_seeds = SeedSequence(1111).spawn(ncores)
    rng_streams1 = [default_rng(s) for s in child_seeds]

    Results = s.parallelize_dataframe(samplePuzzles, rng_streams1, Solver_cols, ncores)
    pd.to_pickle(Results, outfile)

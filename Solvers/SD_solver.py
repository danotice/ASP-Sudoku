import numpy as np
from numpy.random import SeedSequence, default_rng
from copy import deepcopy
import pandas as pd
import multiprocessing as mp
import sudoku as s


#things to change
inpath = './Data/puzzles1000.pkl'
outfile = "./Data/Performance Data/SD_perf.pkl"
maxits = 500000
Runs = 20
ncores = 20
######

subgrids = s.get_Subgrids(3)

subgrid_dict = {}
for i in range(9):
    for j in subgrids[i]:
        subgrid_dict[j] = i


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

def parallelize_dataframe(df, rng_streams, func, n_cores):    

    df_split = np.array_split(df, n_cores)
    pool = mp.Pool(n_cores)
    df = pd.concat(pool.starmap(func, list(zip(df_split,rng_streams))))
    pool.close()
    pool.join()
    return df

def SD_cols(df, rng, lim=maxits,runs=Runs):
    rng = rng

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

    def Swap(soln, sub, a,b):
        #soln input and output in subgrid view
        neighbour = deepcopy(soln)    
        neighbour[sub,a], neighbour[sub,b] = neighbour[sub,b], neighbour[sub,a]        
        return neighbour

     
    def SD_solverIts(empty, init, maxit):
        
        N = len(empty)

        emptySG = to_subgrid_view(empty)
        initSG = to_subgrid_view(init)

        nonfixed = s.GetNonFixedIndSG(emptySG,N)
        
        current = initSG
        f_current = CostFullSG(current,N)
        f_best = f_current
        best_s = deepcopy(current)
        its_best = 0
        its = 0
        #Fs = [f_current]
        
        
        while not f_current == 0 and (its < maxit):
            box = rng.integers(N)
            cand_best_f = np.infty
            best_cand = []

            #find best improvement
            #neighbourhood = [(a,b) for a in nonfixed[box] for b in nonfixed[box] if a != b]
            neighbourhood = [(a,b) for a in nonfixed[box] for b in nonfixed[box] if a > b]
            rng.shuffle(neighbourhood)
            for (a,b) in neighbourhood:   
                #candidate = Insert(current,box,a,b,nonfixed)
                candidate = Swap(current, box,a,b)
                its += 1
                f_new = CostFullSG(candidate,N)

                if f_new < cand_best_f:
                    cand_best_f = f_new
                    best_cand = deepcopy(candidate)
        
            if cand_best_f <= f_current:
                f_current = cand_best_f
                current = deepcopy(best_cand)
                if cand_best_f < f_best:
                    f_best = cand_best_f
                    best_s = deepcopy(current)
                    its_best = its   
        
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


    df[['SD_sols','SD_its','SD_bestIts','SD_obj']] = df.apply(
        lambda row: multiRun_solverM(row.puzzlesF, row.solnInit, SD_solverIts, runs, lim),
        axis=1, result_type='expand')
    return df


if __name__ == "__main__":

    samplePuzzles = pd.read_pickle(inpath)
    #samplePuzzles = samplePuzzles.iloc[:4,:]


    child_seeds = SeedSequence(1111).spawn(ncores)
    rng_streams1 = [default_rng(s) for s in child_seeds]

    Results = parallelize_dataframe(samplePuzzles, rng_streams1, SD_cols, ncores)
    pd.to_pickle(Results, outfile)

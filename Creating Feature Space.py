import numpy as np
from copy import copy

import networkx as nx

from scipy.stats import variation, entropy
from gurobipy import Model, quicksum, GRB

import pandas as pd
import multiprocessing as mp

def format_Puzzle(array,N=9):
    #input to array

    output = array.replace('.','0')
    output = np.array(list(output), dtype=int)
    output = np.reshape(output, (N,N))

    return output

def get_Subgrids(n):
    #list of all subgrid indices
    subgrids = list()
    for I in range(0,n*n,n):
        for J in range(0,n*n,n):
            subgrids.append([(i,j) for i in range(I,I+n) for j in range(J,J+n)])
    return subgrids

#for bygrid=True to work, empty needs to be in row view
def GetFixedCells(empty, bygrid = True):
    n = int(np.sqrt(len(empty)))
    if bygrid:
        fixed = list()
        for I in range(0,n*n,n):
            for J in range(0,n*n,n):
                fixed.append([(i,j) for i in range(I,I+n) for j in range(J,J+n) if empty[i,j]!=0])
    else:
        fixed = [(i,j) for i in range(n*n) for j in range(n*n) if empty[i,j]!=0]
    
    
    return fixed

def GetFixedCellsVal(empty):
    #fixed cell location and value
    n = int(np.sqrt(len(empty)))

    fixed = [(i,j, empty[i,j]) for i in range(n*n) for j in range(n*n) if empty[i,j]!=0]
    return fixed
    
def GetNonFixedCells(empty, bygrid = True):
    n = int(np.sqrt(len(empty)))
    if bygrid:
        nonfixed = list()
        for I in range(0,n*n,n):
            for J in range(0,n*n,n):
                nonfixed.append([(i,j) for i in range(I,I+n) for j in range(J,J+n) if empty[i,j]==0])
    else:
        nonfixed = [(i,j) for i in range(n*n) for j in range(n*n) if empty[i,j]==0]
    
    
    return nonfixed
    

## feature functions
def FixedCellsDist(puzzle, N):

    digitCount = [list(puzzle.flatten()).count(i) for i in range(1,N+1)]
    digitDist = {'num': sum(digitCount),
        'mean': np.mean(digitCount),
        'CV': variation(digitCount),
        'min': np.min(digitCount),
        'max': np.max(digitCount),
        'entropy': entropy(digitCount)
    }
    
    return digitDist

def CountCellStats(puzzle,N):
    #takes empty in row view, nonfixed bygrid=True
        
    nonfixed = GetNonFixedCells(puzzle, True)
    colView = puzzle.transpose()

    countTable = np.zeros((N,N), int)
    
    values = np.zeros(N, int)

    for sub in nonfixed:
        for (I,J) in sub:
            missing = set(range(1,N+1)) - set(puzzle[I]) - set(colView[J]) - {puzzle[i,j] for (i,j) in sub}
            countTable[I,J] = len(missing)            
           
            for v in missing:
                values[v-1] += 1

    countsOut = [i for i in countTable.flatten() if i!=0]
    countStats = {'counts_mean': np.mean(countsOut),
        'counts_naked1': countsOut.count(1),
        'counts_naked2': countsOut.count(2),
        'counts_naked3': countsOut.count(3),
        'counts_CV': variation(countsOut),
        'counts_min': np.min(countsOut),
        'counts_max': np.max(countsOut),
        'counts_entropy': entropy(countsOut),
        'value_mean': np.mean(values),
        'value_CV': variation(values),
        'value_min': np.min(values),
        'value_max': np.max(values),
        'value_entropy': entropy(values)
    }

    return countStats

def emptySets(puzzle,N):
    fixed_G = GetFixedCells(puzzle, True)
    fixed = GetFixedCells(puzzle, False)

    return len([a for a in fixed_G if len(a)==0]) + N-len({i for (i,j) in fixed}) + N-len({j for (i,j) in fixed})


#### SAT features
    
def SATfeatures(puzzle):
    
    def variable_list(N):
        #if z is in cell(x,y) 
        return[(x,y,z) for x in range(1,N+1) for y in range(1,N+1) for z in range(1,N+1)]

    def posclauses_list(N):
        # There is at least one number in each entry
        # all positive literals
        return [tuple([(x,y,z) for z in range(1,N+1)]) for x in range(1,N+1) for y in range(1,N+1)]
        
    def negclauses_list(N,n): #repeats are in this
        # all negative literals
        
        # Each number appears at most once in each row
        n1 = [((x,y,z),(i,y,z)) for y in range(1,N+1) for z in range(1,N+1) for x in range(1,N) for i in range(x+1,N+1)]

        # Each number appears at most once in each column
        n2 = [((x,y,z),(x,i,z)) for x in range(1,N+1) for z in range(1,N+1) for y in range(1,N) for i in range(y+1,N+1)]

        # Each number appears at most once in each 3x3 subgrid
        n3 = [((n*i+x,n*j+y,z),(n*i+x,n*j+k,z)) for z in range(1,N+1) for i in range(n) for j in range(n) for x in range(1,n+1) for y in range(1,n+1) for k in range(y+1,n+1)]
        n3b = [((n*i+x,n*j+y,z),(n*i+k,n*j+l,z)) for z in range(1,N+1) for i in range(n) for j in range(n) for x in range(1,n+1) for y in range(1,n+1) for k in range(x+1,n+1) for l in range(1,n+1)]

        return n1 + n2 + n3 + n3b

    def unitclauses_list(puzzle,N):
        #cells are 0 indexed, output is not
        pos = []
        neg = []
        fixed = GetFixedCells(puzzle, False)
        for (x,y) in fixed:
            z = puzzle[x,y]
            vals = list(range(1,N+1))
            vals.remove(z)
            pos = pos + [((x+1,y+1,z),)]
            neg = neg + [((x+1,y+1,i),) for i in vals]
        
        return pos, neg  

    def Clause_list(puzzle):

        N = len(puzzle)
        n = int(np.sqrt(N))
        posU, negU = unitclauses_list(puzzle,N)
        
        return list(dict.fromkeys(posclauses_list(N) + posU)), list(dict.fromkeys(negclauses_list(N,n) + negU))

    def CG_features(clausesp, clausesn, vars):
    
        CG = nx.Graph()

        CG.add_nodes_from([str(nd) for nd in clausesn])
        CG.add_nodes_from([str(nd) for nd in clausesp])

        for nd in vars:
            tmpSet = [str(cl) for cl in clausesn if nd in cl]
            L = len(tmpSet)
            if L > 1:
                for i in range(L-1):
                    for j in range(i+1, L):
                        CG.add_edge(tmpSet[i],tmpSet[j])

        degCG = list(dict(CG.degree()).values())

        cgNodesDeg = {'CG_mean': np.mean(degCG),
            'CG_CV': variation(degCG),
            'CG_min': np.min(degCG),
            'CG_max': np.max(degCG),
            'CG_entropy': entropy(degCG)
        }

        clust = list(dict(nx.clustering(CG)).values())
        clusNodesDeg = {'CGclust_mean': np.mean(clust),
            'CGclustCV': variation(clust),
            'CGclustmin': np.min(clust),
            'CGclustmax': np.max(clust),
            'CGclustentropy': entropy(clust)
        }

        return cgNodesDeg, clusNodesDeg
                    
    def VG_features(clausesp, clausesn, vars):
        VG = nx.Graph()

        VG.add_nodes_from([str(nd) for nd in vars])
        
        for nd in vars:
            tmpSet = [str(cl) for cl in clausesn+clausesp if nd in cl]
            L = len(tmpSet)
            if L > 1:
                for i in range(L-1):
                    for j in range(i+1, L):
                        VG.add_edge(tmpSet[i],tmpSet[j])

        degVG = list(dict(VG.degree()).values())

        vgNodesDeg = {'VG_mean': np.mean(degVG),
            'VG_CV': variation(degVG),
            'VG_min': np.min(degVG),
            'VG_max': np.max(degVG),
            'VG_entropy': entropy(degVG)
        }
    

        return vgNodesDeg

    def LPrelax(clausesp, clausesn, vars):
    
        model = Model()
        model.setParam("OutputFlag",0)

        x = model.addVars(vars, lb=0,ub=1)

        for cl in clausesp:
            model.addConstr(quicksum(x[l] for l in cl) >= 1.0)

        for cl in clausesn:
            model.addConstr(quicksum(1-x[l] for l in cl) >= 1.0)

        model.setObjective(quicksum(x[l] for l in cl for cl in clausesp) + quicksum(1-x[l] for l in cl for cl in clausesp), GRB.MAXIMIZE)
        model.optimize()

        lpStats = {
            'LP_obj': model.ObjVal, 'LP_fracInt': len([x for x in model.X if x == 1 or x==0])/model.NumVars
        }

        slack = model.Slack

        slackStats = {'LPslack_mean': np.mean(slack),
            'LPslack_CV': variation(slack),
            'LPslack_min': np.min(slack),
            'LPslack_max': np.max(slack),
            'LPslack_entropy': entropy(slack)
        }

        return lpStats, slackStats

    N = len(puzzle)
    
    clausesp, clausesn = Clause_list(puzzle)
    vars = variable_list(N)

    c = len(clausesp+clausesn)
    v = len(vars)
    ratio = c/v

    sizeFeatures = {
        'SAT_c': c, 
        'SAT_v': v, 
        'SAT_ratio': ratio, 'SAT_ratio2': ratio**2, 'SAT_ratio3': ratio**3,
        'SAT_ratioRec': 1/ratio, 'SAT_ratioRec2': 1/ratio**2, 'SAT_ratioRec3': 1/ratio**3,
        'SAT_ratioLin': abs(4.26 - ratio), 'SAT_ratioLin2': abs(4.26 - ratio)**2, 'SAT_ratioLin3': abs(4.26 - ratio)**3
        }


    vgFeatures = VG_features(clausesp, clausesn, vars)
    cgNodesDeg, clusNodesDeg = CG_features(clausesp, clausesn, vars)
    lpStats, slackStats = LPrelax(clausesp, clausesn, vars)

    return sizeFeatures | vgFeatures| cgNodesDeg| clusNodesDeg| lpStats| slackStats


#### GCP features

def PuzzletoGCPfeatures(puzzle, N):

    def PuzzletoGCP(puzzle, N):
        n = int(np.sqrt(N))
        
        G = nx.relabel_nodes(nx.sudoku_graph(n), {i: (col,i) for (i,col) in enumerate(puzzle.flatten())}, copy=False)
        G_ = copy(G)

        nodeGroups = {i : [cell for cell in list(G.nodes) if cell[0]==i] for i in range(1,N+1)}

        for i in range(1, N+1):
            while len(nodeGroups[i]) > 1:
                nd1 = nodeGroups[i].pop()
                nd2 = nodeGroups[i].pop()
                G_ = nx.contracted_nodes(G_, nd1,nd2, self_loops=False)

                nodeGroups[i] = [cell for cell in list(G_.nodes) if cell[0]==i]

        contracted = [a for l in list(nodeGroups.values()) for a in l]
        for i in range(len(contracted)):
            for j in range(i+1, len(contracted)):
                G_.add_edge(contracted[i],contracted[j])

        return G, G_

    _, G_ = PuzzletoGCP(puzzle, N)
    
    deg = list(dict(G_.degree()).values())
    bc = list(dict(nx.betweenness_centrality(G_)).values())

    return {
        'GCP_nodes': len(G_.nodes),
        'GCP_edges': len(G_.edges),
        'GCP_density': nx.density(G_),
        'GCP_nDeg_mean': np.mean(deg),
        'GCP_nDeg_std': np.std(deg),
        'GCP_avgPath': nx.average_shortest_path_length(G_),
        'GCP_diameter': nx.diameter(G_),
        'GCP_bc_mean': np.mean(bc),
        'GCP_bc_std': np.std(bc),
        'GCP_clustcoef': nx.average_clustering(G_),
        'GCP_weiner': nx.wiener_index(G_)
    }

def parallelize_dataframe(df, func, n_cores=4):    

    df_split = np.array_split(df, n_cores)
    pool = mp.Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

## loading data

path = './Data/puzzles1000.pkl'
samples = pd.read_pickle(path)
samples = samples.iloc[:4,:5] #to exclude initial solns

### to create feature space
def create_feature_space(samplePuzzles):
    #formatting puzzles
    samplePuzzles['N'] = 9

    #fixed cell distribution
    addcols = ['fixedDig_num','fixedDig_mean','fixedDig_CV',
        'fixedDig_min','fixedDig_max','fixedDig_entropy']

    samplePuzzles[addcols] = samplePuzzles.apply(
        lambda row: FixedCellsDist(row.puzzlesF, row.N),
        axis=1, result_type='expand')

    addcols = ['counts_mean', 'counts_naked1', 'counts_naked2', 'counts_naked3',
        'counts_CV', 'counts_min', 'counts_max', 'counts_entropy', 'value_mean',
        'value_CV', 'value_min', 'value_max', 'value_entropy']

    samplePuzzles[addcols] = samplePuzzles.apply(
        lambda row: CountCellStats(row.puzzlesF, row.N),
        axis=1, result_type='expand')

    samplePuzzles['emptySets'] = samplePuzzles.apply(
        lambda row: emptySets(row.puzzlesF, row.N),
        axis=1)

    #SAT features
    addcols = ['SAT_c', 'SAT_v', 'SAT_ratio', 'SAT_ratio2', 'SAT_ratio3',
        'SAT_ratioRec', 'SAT_ratioRec2', 'SAT_ratioRec3', 'SAT_ratioLin',
        'SAT_ratioLin2', 'SAT_ratioLin3', 'VG_mean', 'VG_CV', 'VG_min',
        'VG_max', 'VG_entropy', 'CG_mean', 'CG_CV', 'CG_min', 'CG_max',
        'CG_entropy', 'CGclust_mean', 'CGclustCV', 'CGclustmin', 'CGclustmax',
        'CGclustentropy', 'LP_obj', 'LP_fracInt', 'LPslack_mean', 'LPslack_CV',
        'LPslack_min', 'LPslack_max', 'LPslack_entropy']

    samplePuzzles[addcols] = samplePuzzles.apply(
        lambda row: SATfeatures(row.puzzlesF),
        axis=1, result_type='expand')

    addcols = ['GCP_nodes', 'GCP_edges', 'GCP_density', 'GCP_nDeg_mean', 
        'GCP_nDeg_std', 'GCP_avgPath', 'GCP_diameter', 'GCP_bc_mean', 'GCP_bc_std', 
        'GCP_clustcoef', 'GCP_weiner']


    samplePuzzles[addcols] = samplePuzzles.apply(
        lambda row: PuzzletoGCPfeatures(row.puzzlesF, row.N),
        axis=1, result_type='expand')

    return samplePuzzles

samples = parallelize_dataframe(samples, create_feature_space,4)


#output
samples.to_pickle('./Data/featureSpace1000.pkl')

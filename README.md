# The Algorithm Selection Problem for solving Sudoku with metaheuristics

This repository contains the data and code for the work on this paper to be presented at IEEE 2023 Congress on Evolutionary Computation.

In this work we study the algorithm selection problem and instance space analysis  for solving Sudoku puzzles with metaheuristic algorithms. 
We formulate Sudoku as a combinatorial optimisation problem and implement four local-search metaheuristics to solve the problem instances. 
The aim is to use ISA to determine how these features affect the performance of the algorithms and how they can be used for automated algorithm selection.
We also consider algorithm selection using multinomial logistic regression models with $l_1$-penalty for comparison.

Included are:
- Python scripts to create the feature space described. (\* *A Gurobi licences is required to extract full set of SAT reformulation features.*)
- Python scripts for each of the 4 algorithms used to solve the puzzles.
- Feature and performance data for 1000 Sudoku puzzles (further described in Data folder).
- Jupyter notebooks to generate the results discussed in the conference paper.
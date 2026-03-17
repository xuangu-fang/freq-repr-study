# Task Templates

## Template A: Build dataset
Goal:
- implement or validate one demo dataset generator

Steps:
1. read DEMO_SPECS.md
2. implement generator
3. generate 5 sample trajectories
4. save preview plots
5. report shape and metadata

## Template B: Add representation
Goal:
- implement one representation module

Steps:
1. read REPRESENTATIONS.md
2. implement fit/transform API
3. test on one trajectory from each demo
4. save transformed shape and metadata
5. update registry if needed

## Template C: Add metric
Goal:
- implement one metric module

Steps:
1. read METRICS.md
2. implement metric API
3. test on synthetic representation data
4. run on one real trajectory
5. save aggregate output example

## Template D: Run benchmark
Goal:
- run one benchmark config end-to-end

Steps:
1. generate/load dataset
2. extract representations
3. compute metrics
4. save figures and summary
5. report failures if any
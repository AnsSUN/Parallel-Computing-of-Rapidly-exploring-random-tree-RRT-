# Parallel-Computing-of-Rapidly-exploring-random-tree-RRT-
RRT mostly use in Robotics to get the optimal path in configuration space. This project tries to achieve less computation time through parallel computing

Numba was used to for cuda programming in python3.
Moreover it use the potential of multithreading in normal processor along with the Graphical Processing Unit (GPU). Dask package is used for mutithreading for processing.

This project is very good example of using parallel computing on cores of both processors i.e. GPU and microprocessor.
This project also compares the different density of RRT generated at different amount of time to get the performance of parallel computing.

Many obstacle is also placed in the configuration space (Square) to get the optimal path without hitting them.

Use command: "python -m tbb parallel_RRT_test_Dask_numba.py"

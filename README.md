This is an implementation of M2HL which can guarantee the minimality and correctness properties of 2-hop labeling index in the edge insertion and deletion scenarios. The datasets can be downloaded from "http://konect.cc/networks/", "https://networkrepository.com/index.php", and "https://law.di.unimi.it/".

Dataset format:
n m       // the number of nodes and edges
v1 v2 v3  // the id of v0's neighbors
v0 v2     // the id of v1's neighbors
v0 v1     // the id of v2's neighbors
v0        // the id of v3's neighbors

Input parameters:
    Graphpath : the path of original graph G
    
    Indexpath : the path of the 2-hop labeling of G
    
    program_choice : the execute program
        0:    Execute PSL on G
        1:    Execute M2HL (edge deletion)
        2:    Execute PSL  (edge deletion)
        3:    Execute M2HL (edge insertion)
        4:    Execute PSL  (edge insertion)
        rest: Execute the 2-hop labeling-based query
    
    active_ratio : the places of the activated vertices (0, 1)

    threads : the number of threads during the parallel computing
    
    dynamic_edge_num : the number of dynamic edges 

    query_task_num : the number of query tasks

Output:
    
    program 0: the 2-hop index of G
    
    program 1: the updated 2-hop index and processing time (edge deletion)
    
    program 2: the reconstructed 2-hop index and processing time (edge deletion)
    
    program 3: the updated 2-hop index and processing time (edge insertion)
    
    program 4: the reconstructed 2-hop index and processing time (edge insertion)
    
    rest     : the average query time

Compile: 
    g++ -O3 -std=c++11 -fopenmp IndexC.cpp -o run

Run example:

    ./run "test.graph" "test.bin" 0 0.5 40 1000 100000 // build the 2-hop index of test.graph

    ./run "test.graph" "test.bin" 1 0.5 40 1000 100000 // Update the 2-hop index of test.graph (edge deletion)

    ./run "test.graph" "test.bin" 3 0.5 40 1000 100000 // Update the 2-hop index of test.graph (edge insertion)

    ./run "test.graph" "test.bin" 5 0.5 40 1000 100000 // Execute 100000 query tasks on test.graph

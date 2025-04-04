# Fast Optimal Group Steiner Tree Search using GPUs

## GST_data
The dataset of the paper is stored on [OneDrive](https://1drv.ms/f/c/683d9dd9f262486b/Ek6Fl_brQzhDnI2cmhGIHxMBQ-L1ApeSqxwZKE4NBsDXSQ?e=3RBc8S). There are eight datasets: Twitch, Musae, Github,  Youtube, Orkut, DBLP, Reddit, LiveJournal. There are 8 files for each dataset. For example, the Twitch dataset contains the following 8 files:
1. "Twitch.in". This readable file contains the basic information of this dataset. The two numbers on the first line of the file represent the number of vertices and edges in the graph. Each of the following lines contains three numbers representing the two end vertices and the weight of an edge. For example, "18 14919 100" shows that there is an edge between vertex 18 and vertex 14919, with an edge weight of 100.

2. "Twitch_beg_pos.bin". This is a binary file. It contains V elements, each element representing the starting position of a vertex's adjacency list. Therefore, the position of a vertex can be obtained by subtracting the starting position of the next vertex from the starting position of that vertex.

3. "Twitch_csr.bin". This is a binary file. It contains E elements, and this file stores an adjacency list of vertices, where each element represents an endpoint of an edge.

4. "Twitch_weight.bin". This is a binary file. It contains E elements, which store the weights of edges, with each element representing the weight of an edge.

5. "Twitch.g". Each line of this file represents which vertices in the graph are included in a group. For example, "g7:2705 13464 16088 16341 22323" indicates that group 7 contains five vertices: 2705, 13464, 16088, 16341, and 22323.

6. "Twitch3.csv". Each line of this file represents a query of size 3. For example, "2475 2384 159" indicates that the return tree of this query must contain group 2475, 2384, and 159.

7. "Twitch5.csv". Each line of this file represents a query of size 5. For example, "1016 2941 1613 1105 2228" indicates that the return tree of this query must contain group 1016, 2941, 1613, 1105, 2228.

8. "Twitch7.csv". Each line of this file represents a query of size 7. For example, "3137 393 742 25 2125 2122 727" indicates that the return tree of this query must contain group 3137, 393, 742, 25, 2125, 2122, 727.

   
## Running code example
Here, we show how to build and run experiments on a Linux server with the Ubuntu 20.04 system, an Intel(R) Xeon(R) Platinum 8360Y CPU @ 2.40GHz, and 1 NVIDIA GeForce RTX A6000 GPU. The environment is as follows:
- gcc version 9.3.0 (GCC)
- CUDA compiler NVIDIA 11.8.89
- cmake version 3.28.3
- Boost
We will provide a detailed introduction to the experimental process as follows.

In your appropriate directory, execute the following commands:

Download the code:
```
git clone https://anonymous.4open.science/r/GPU4GST/
```
Switch the working directory to GPU4GST.
```
cd GPU4GST
```
Download the dataset from [OneDrive](https://1drv.ms/f/c/683d9dd9f262486b/Ek6Fl_brQzhDnI2cmhGIHxMBQ-L1ApeSqxwZKE4NBsDXSQ?e=3RBc8S). Assume that the dataset is located in the "data" folder of the working directory GPU4GST.



After preparing the environment according to the above suggestions, we can use the sh files in the "sh" folder to compile and run the code.
Among them, example.sh conducts experiments on six algorithms using the Twitch dataset, with each algorithm executing 50 queries of size 3. The running instruction is:
 ```
sh sh/example.sh
 ```
The experiment results will be automatically saved as CSV files in the "data/result" folder.


The other six sh files correspond to complete experiments of an algorithm on eight datasets, with 300 queries of sizes 3, 5, and 7 executed on each dataset. For example, to run experiments for TrimCDP-WB, using the following instruction:

 ```
sh sh/exp_TrimCDP-WB.sh
 ```


Taking exp_D-TrimCDP-WB.sh as an example:
```
cd code/D-TrimCDP-WB/build
mkdir build
cd build
cmake ..
make
```
The above instructions switch to the corresponding directory of the algorithm and compile the code into an executable file.
```
./bin/D-TrimCDP-WB 2 ../../../data/ Musae 3 4 0 299
```
This instruction executes the executable file, specifying the query size, the dataset to be used and its location, the upper bound of the diameter constraint, and the start and end indices of the queries to be executed.


## GST_code
All codes are located in the 'code' folder. There are six subfolders, each corresponding to codes of one of six experiments in the paper.
- PrunedDP++. This is the PrunedDP++ version code of GST without diameter constraints.
- TrimCDP. This is the TrimCDP version code without diameter constraint for GST.
- TrimCDP-WB. This is the TrimCDP-WB version code without diameter constraint for GST.
- D-PrunedDP++. This is the PrunedDP++ version code with diameter constraints for GST.
- D-TrimCDP. This is the TrimCDP version code with diameter constraints for GST.
- D-TrimCDP-WB. This is the TrimCDP-WB version code with diameter constraints for GST.

In the six subfolders, there are .h, .cu, .cuh, and .cpp files used for conducting experiments in the paper. The .h and .cuh files are in the "include" directory, while the .cpp files are in the "src" directory. Some examples are as follows.


### CPU:
- "PrunedDP++/src/main.cpp" contains codes for conducting experiments for PrunedDP++. 
- "PrunedDP++/include/CPUNONHOP.h" contains codes of PrunedDP++.


### GPU
- "TrimCDP/src/main.cpp" contains codes for conducting experiments for TrimCDP. 
- "TrimCDP/include/exp_GPU_nonHop.h" contains code for reading the graph, groups, and queries.
- "TrimCDP/src/DPQ.cu" contains codes of TrimCDP.


### GPU+
- "TrimCDP-WB/src/GSTnonHop.cu" contains codes for conducting experiments for TrimCDP-WB. 
- "TrimCDP-WB/include/mapper_enactor.cuh" contains the overall framework of TrimCDP-WB.
- "TrimCDP-WB/include/mapper.cuh" contains codes for performing specific operations on vertices, such as grow and merge operations.
- "TrimCDP-WB/include/reducer.cuh" contains codes for organizing and allocating work after completing vertices operations.


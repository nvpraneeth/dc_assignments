Vector Clock and Singhal-Kshemkalyani Optimization Implementation
================================================================

This project implements and compares two distributed clock synchronization algorithms:
1. Basic Vector Clock (VC-CS25RESCH04001.cpp)
2. Singhal-Kshemkalyani Optimization (SK-CS25RESCH04001.cpp)

Project Structure:
-----------------
- VC-CS25RESCH04001.cpp          : Basic Vector Clock implementation
- SK-CS25RESCH04001.cpp          : Singhal-Kshemkalyani optimization
- performance_comparison.cpp : Performance testing script
- generate_graphs.py         : Graph generation script
- Makefile                   : Build automation
- inp-params.txt            : Input parameters file
- readme.txt                : This file
- project_report.pdf        : Comprehensive project report

Compilation Instructions:
------------------------
1. Compile Vector Clock implementation:
   g++ -std=c++17 -pthread -o VC-CS25RESCH04001 VC-CS25RESCH04001.cpp

2. Compile Singhal-Kshemkalyani implementation:
   g++ -std=c++17 -pthread -o SK-CS25RESCH04001 SK-CS25RESCH04001.cpp

3. Compile performance comparison script:
   g++ -std=c++17 -pthread -o performance_comparison performance_comparison.cpp

4. Or use Makefile for easy compilation:
   make all

Execution Instructions:
----------------------
1. Basic Execution:
   - Ensure inp-params.txt is present with correct parameters
   - Run: ./VC-CS25RESCH04001 (for Vector Clock)
   - Run: ./SK-CS25RESCH04001 (for Singhal-Kshemkalyani)

2. Performance Comparison:
   - Run: ./performance_comparison
   - This will test both algorithms with varying process counts (10-15)
   - Results will be saved to performance_results.txt and performance_data.csv

3. Graph Generation:
   - Install Python dependencies: pip install pandas matplotlib seaborn numpy
   - Run: python3 generate_graphs.py
   - This generates performance_comparison_graphs.png and analysis_report.txt

4. Using Makefile:
   - make run-vc     (run Vector Clock)
   - make run-sk     (run Singhal-Kshemkalyani)
   - make run-perf   (run performance comparison)
   - make graphs     (generate graphs)

Input File Format (inp-params.txt):
-----------------------------------
Line 1: n λ α m
- n: number of processes
- λ: exponential distribution parameter (ms)
- α: ratio of internal to send events
- m: maximum messages per process

Lines 2-n+1: Graph topology (adjacency list)
- Each line represents neighbors of process i

Example:
3 5 1.5 40
1 2 3
2 1 3
3 1 2

Output Files:
-------------
- vector_clock_log.txt: Event log for Vector Clock implementation
- singhal_kshemkalyani_log.txt: Event log for SK implementation
- performance_results.txt: Detailed performance comparison
- performance_data.csv: Data for graph generation
- performance_comparison_graphs.png: Generated graphs
- analysis_report.txt: Detailed analysis report

Algorithm Details:
------------------
1. Vector Clock (VC-CS25RESCH04001.cpp):
   - Implements basic vector clock algorithm
   - Sends full vector clock with every message
   - Demonstrates strong consistency property

2. Singhal-Kshemkalyani (SK-CS25RESCH04001.cpp):
   - Optimized version that sends only changed entries
   - Maintains last_sent_clock to track what was sent
   - Reduces message overhead significantly

Event Types:
------------
- Internal events: Process-local operations
- Send events: Message transmission to neighbors
- Receive events: Message reception from neighbors

Performance Metrics:
--------------------
- Average entries per message
- Total messages sent
- Execution time
- Space utilization per process

System Requirements:
--------------------
- C++17 compatible compiler (g++ recommended)
- pthread library
- Python 3.x (for graph generation)
- Python packages: pandas, matplotlib, seaborn, numpy

Troubleshooting:
----------------
1. Compilation errors: Ensure C++17 support and pthread library
2. Runtime errors: Check inp-params.txt format
3. Graph generation errors: Install required Python packages
4. Permission errors: Ensure execute permissions on compiled binaries


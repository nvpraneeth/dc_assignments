===============================================================================
DISTRIBUTED COMPUTING - PROGRAMMING ASSIGNMENT 3
Maekawa's Algorithm and Ricart-Agarwala's Algorithm Implementation
===============================================================================

AUTHOR: CS25RESCH04001

===============================================================================
FILES INCLUDED
===============================================================================

1. MK-CS25RESCH04001.cpp   - Implementation of Maekawa's Algorithm with Grid Quorum
2. RA-CS25RESCH04001.cpp   - Implementation of Ricart-Agarwala's Algorithm
3. common.h                 - Common header file with shared data structures
4. common.cpp               - Common utility functions
5. inp-params.txt          - Input parameters file
6. Makefile                 - Makefile for compilation
7. readme.txt              - This file

===============================================================================
COMPILATION INSTRUCTIONS
===============================================================================

1. Compile the Maekawa's Algorithm program:
   make mk
   
   Or manually:
   g++ -std=c++11 -pthread MK-CS25RESCH04001.cpp common.cpp -o mk-node

2. Compile the Ricart-Agarwala's Algorithm program:
   make ra
   
   Or manually:
   g++ -std=c++11 -pthread RA-CS25RESCH04001.cpp common.cpp -o ra-node

3. Compile both:
   make all

4. Clean compiled files:
   make clean

===============================================================================
INPUT FILE FORMAT
===============================================================================

The inp-params.txt file should contain exactly one line with the following format:
n k alpha beta

Where:
- n      : Number of processes (must be a perfect square for Maekawa's algorithm)
- k      : Number of times each process enters the critical section
- alpha  : Average time for local computation (exponential distribution)
- beta   : Average time inside critical section (exponential distribution)
          (Note: alpha < beta as per problem statement)

Example:
9 15 2.0 5.0

This means:
- 9 processes (3x3 grid for Maekawa)
- Each process enters CS 15 times
- Average local computation time: 2.0 seconds
- Average time in CS: 5.0 seconds

===============================================================================
EXECUTION INSTRUCTIONS
===============================================================================

MAEKawa's ALGORITHM:

1. Open multiple terminal windows (one for each process). For n processes, you need n terminals.

2. In each terminal, navigate to the project directory:
   cd <path_to_project_directory>

3. Run the program in each terminal with the process ID:
   Terminal 0: ./mk-node 0
   Terminal 1: ./mk-node 1
   Terminal 2: ./mk-node 2
   ...
   Terminal n-1: ./mk-node <n-1>

4. Or use a shell script to launch all processes:
   chmod +x run-mk.sh
   ./run-mk.sh

   (You may need to create this script)

RICART-AGARWALA'S ALGORITHM:

1. Similar to Maekawa's algorithm, open n terminal windows.

2. Run the program in each terminal with the process ID:
   Terminal 0: ./ra-node 0
   Terminal 1: ./ra-node 1
   Terminal 2: ./ra-node 2
   ...
   Terminal n-1: ./ra-node <n-1>

3. Or use a shell script:
   chmod +x run-ra.sh
   ./run-ra.sh

===============================================================================
OUTPUT FILES
===============================================================================

For each process pi, the following files are generated:

1. pi.log - Log file containing all message exchanges for process i
   Example: p0.log, p1.log, p2.log, etc.

2. mk_stats.txt (for Maekawa) - Statistics file from coordinator (process 0)
3. ra_stats.txt (for Ricart-Agarwala) - Statistics file from coordinator (process 0)

Log file format:
Each log file contains timestamps and message exchanges, for example:
p1 is doing local computation at 10:00:00.123
p1 requests to enter CS at 10:00:02.456 for the 1st time
p1 receives p2's request to enter CS at 10:00:03.789
p1 replies to p2's request to enter CS at 10:00:03.890
p1 enters CS at 10:00:05.234 for the 1st time
p1 leaves CS at 10:00:10.567 for the 1st time

===============================================================================
EXPERIMENT SETUP
===============================================================================

For Experiment 1 (Message Complexity):
- Vary n from 5 to 25 (use perfect squares: 9, 16, 25, etc.)
- Keep k constant at 15
- Run each configuration 5 times and average the results
- Collect total message count from coordinator

For Experiment 2 (Time Complexity):
- Vary n from 5 to 25 (use perfect squares: 9, 16, 25, etc.)
- Keep k constant at 15
- Run each configuration 5 times and average the results
- Collect average wait time from coordinator

===============================================================================
SYSTEM REQUIREMENTS
===============================================================================

- Linux/Unix environment (tested on Ubuntu/CentOS)
- g++ compiler with C++11 support
- pthread library (usually included with g++)
- Sufficient port range available (default: ports 10000+ for MK, 20000+ for RA)

===============================================================================
TROUBLESHOOTING
===============================================================================

1. "Address already in use" error:
   - Wait a few seconds between runs for ports to be released
   - Or change port_base in the source code

2. "Connection refused" error:
   - Make sure all processes start in the correct order
   - Add more delay between process starts if needed

3. Processes hanging:
   - Check if all processes are running
   - Verify inp-params.txt is correct
   - Check log files for errors

4. Compilation errors:
   - Ensure C++11 support: g++ --version
   - Check if pthread is available
   - Verify all source files are present

===============================================================================
NOTES
===============================================================================

- For Maekawa's algorithm, n must be a perfect square (4, 9, 16, 25, etc.)
- Processes communicate via TCP sockets on localhost
- Process 0 acts as the coordinator for statistics collection
- The program automatically cleans up connections on termination
- All times are in seconds (converted to milliseconds for sleep)

===============================================================================

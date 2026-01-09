================================================================================
Federated Learning Simulation - Execution Instructions
================================================================================

PROJECT STRUCTURE:
------------------
- utils.py: Common utilities (data loading, LeNet-5 model, hybrid clock)
- fedavg.py: Centralized Federated Learning (Protocol 0)
- fedprox.py: Centralized Federated Learning with Proximal Regularization (Protocol 5)
- decentralized_baseline.py: Decentralized Baseline with Random Peer Selection (Protocol 1)
- mpls_list_scheduling.py: MPLS with List Scheduling Algorithm
- mpls_handshake_improved.py: MPLS with Improved Asynchronous Handshake
- mpls_handshake_improved_synchronized.py: MPLS with Improved Synchronous Handshake
- experiments.py: Experiment runner that executes all protocols and generates graphs
- inp-params.txt: Input parameters file

Note: There are many other protocol implementations available in this folder (e.g., 
mpls_handshake.py, mpls_deterministic.py, mpls_handshake_momentum.py, decentralized_handshake.py, 
etc.). Check the folder for additional variants and implementations.

INPUT FILE FORMAT (inp-params*.txt):
------------------------------------
- First line is a comment describing each field (ignored by the code)
- Next non-empty line contains comma-separated values:
  N,R_max,E,B,Protocol,C,MU

Where:
  N: Total number of processes/nodes
  R_max: Maximum communication rounds
  E: Local training epochs per round
  B: Link Bandwidth in Mbps (Megabits per second)
  Protocol: 0=FedAvg, 1=Decentralized Baseline, 2=MPLS, 5=FedProx
  (Note: Protocol value is mainly used by experiments.py; individual scripts 
   may use their own input parameter files)
  C: Candidate set size (peer selection fanout)
  MU: FedProx proximal coefficient (ignored by other protocols)

Example:
  # Columns: N,...,MU
  10,50,5,10.0,0,5,0.1
  (10 nodes, 50 rounds, 5 epochs, 10 Mbps, Protocol 0, candidate size 5, mu=0.1)

Convenience files (`inp-fedavg.txt`, `inp-fedprox.txt`, `inp-decentralized.txt`, `inp-mpls.txt`) are provided with sensible defaults per protocol; copy or edit them as needed before running a script.

INSTALLATION:
-------------
1. Install required Python packages:
   pip install torch torchvision matplotlib numpy

2. Ensure Python 3.7+ is installed

EXECUTION:
----------

Option 1: Run Individual Protocol
----------------------------------
To run a specific protocol, use the appropriate input parameter file or modify inp-params.txt
and execute:

  python fedavg.py                              # FedAvg (Centralized)
  python fedprox.py                             # FedProx (Centralized with Proximal Term)
  python mpls_list_scheduling.py                # MPLS with List Scheduling Algorithm
  python mpls_handshake_improved.py              # MPLS with Improved Asynchronous Handshake
  python mpls_handshake_improved_synchronized.py # MPLS with Improved Synchronous Handshake
  python decentralized_baseline.py               # Decentralized Baseline (Random peer selection)

Note: There are many other protocol implementations available in this folder. Check the folder
for additional variants such as mpls_handshake.py, mpls_deterministic.py, 
mpls_handshake_momentum.py, decentralized_handshake.py, etc.

Logs will be saved in:
  - logs/fedavg/ or logs/fedavg_run*/ (for FedAvg)
  - logs/fedprox/ or logs/fedprox_run*/ (for FedProx)
  - logs/mpls_list_scheduling/ (for MPLS List Scheduling)
  - logs/mpls_handshake_improved/ (for MPLS Improved Handshake)
  - logs/mpls_handshake_improved_synchronized/ (for MPLS Improved Synchronized)
  - logs/decentralized/ or logs/decentralized_run*/ (for Decentralized Baseline)

Option 2: Run All Experiments (Recommended)
-------------------------------------------
To run all three protocols multiple times and generate comparative graphs:

  python experiments.py [num_runs]

  Default: 5 runs per protocol
  Example: python experiments.py 3  (runs 3 times per protocol)

This will:
  1. Run each protocol/variant num_runs times
  2. Average the results
  3. Generate 4 comparative graphs in experiments_output/:
     - exp1_accuracy_vs_rounds.png
     - exp2_accuracy_vs_virtual_time.png
     - exp3_accuracy_vs_data_volume.png
     - exp4_messages_vs_rounds.png
  4. Save results summary in experiments_output/results_summary.txt

Note: When running experiments.py, the Protocol value in inp-params.txt is ignored
      as all three protocols are executed.

OUTPUT:
-------
Each process logs to individual log files (p0_log.txt, p1_log.txt, etc.) showing:
  - Round progress
  - Compute time measurements
  - Network delay calculations
  - Communication details
  - Accuracy at each round

The coordinator process (rank 0) collects global statistics.

HYBRID CLOCK MECHANISM:
-----------------------
The implementation uses a Hybrid Clock that tracks:
  - Real Compute Time: Measured using time.process_time() for actual CPU time
  - Simulated Network Time: Calculated as payload_size / bandwidth

Virtual Time = Sum of (Compute Time + Network Delay)

This ensures accurate time measurements in a single-machine simulation.

NON-IID DATA DISTRIBUTION:
--------------------------
The system automatically creates Non-IID data distribution:
  - Each node receives data from only 2 classes (label skew)
  - Classes are distributed across nodes to create heterogeneity
  - This demonstrates the robustness of smart peer selection in MPLS

TROUBLESHOOTING:
---------------
1. If you encounter "spawn" method errors:
   - The code uses torch.multiprocessing with 'spawn' method
   - Ensure all imports are at the top level of modules

2. If processes hang:
   - Check that all processes can access shared memory
   - Ensure sufficient system resources

3. If MNIST download fails:
   - Check internet connection
   - The dataset will be downloaded to ./data/ directory

4. For faster execution during testing:
   - Reduce R_max (e.g., 10 instead of 50)
   - Reduce N (e.g., 5 instead of 10)
   - Reduce E (e.g., 1 instead of 5)

PERFORMANCE NOTES:
------------------
- Running with N=10, R_max=50, E=5 may take 30-60 minutes depending on hardware
- Each protocol run is independent and can be parallelized manually if needed
- The experiments.py script runs protocols sequentially for accurate comparison

================================================================================


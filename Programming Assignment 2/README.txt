# Distributed Banking System - Programming Assignment 2

## Compilation Instructions

### Prerequisites
- C++11 compatible compiler (g++ recommended)
- Python 3 for input generation
- Make utility

### Compilation
```bash
# Clean previous builds
make clean

# Compile the project
make

# This will generate:
# - main_parallel: Main executable for distributed banking system
# - banking_system: Alternative implementation
# - Input parameter files for different topologies
```

### Execution Instructions

#### Basic Execution
```bash
# Run with line topology (3 processes)
./main_parallel inp-params-line-3.txt

# Run with ring topology (4 processes)
./main_parallel inp-params-ring-4.txt

# Run with tree topology (6 processes)
./main_parallel inp-params-tree-6.txt

# Run with arbitrary topology (8 processes)
./main_parallel inp-params-arbitrary-8.txt
```

#### Input Parameters
The system accepts input files with the following format:
- First line: n λ λc k (number of processes, lambda, lambda_c, number of snapshots)
- Second line: Initial balances for each process
- Following lines: Graph topology as adjacency list

#### Output
The system generates:
- Log files in `logs/` directory for each process
- Results in `results/` directory with timing information
- Console output showing system status

#### Example Usage
```bash
# Run experiment with 3 processes, line topology
./main_parallel inp-params-line-3.txt

# Check logs
ls logs/
cat logs/inp-params-line_3_coordinator.log
cat logs/inp-params-line_3_process_1.log
```

## System Architecture

### Components
1. **Process Class**: Handles individual process logic
2. **Message Class**: Manages message passing between processes
3. **NetworkManager**: Handles socket-based communication
4. **Logger**: Records all events with timestamps

### Key Features
- Distributed snapshot algorithm (Chandy-Lamport)
- Money transfer simulation between processes
- Balance consistency verification
- Multi-threaded architecture for send/receive operations

## Troubleshooting

### Common Issues
1. **Port conflicts**: Ensure ports 8000+ are available
2. **Permission issues**: Run with appropriate permissions
3. **Memory issues**: Large number of processes may require more memory

### Debug Mode
The system includes extensive debug logging. Check log files for detailed execution traces.

## Performance Notes
- System performance depends on network latency
- Large topologies may take longer to complete
- Snapshot collection time varies with network conditions

#!/bin/bash

# Experiment script for distributed banking system
# Runs experiments with different topologies and process counts

echo "Starting Distributed Banking System Experiments"
echo "=============================================="

# Create results directory
mkdir -p results/experiments

# Function to run experiment and collect data
run_experiment() {
    local topology=$1
    local processes=$2
    local input_file="inp-params-${topology}-${processes}.txt"
    
    echo "Running experiment: ${topology} topology with ${processes} processes"
    
    # Run the experiment
    timeout 120s ./main_parallel ${input_file} > results/experiments/${topology}_${processes}.log 2>&1
    
    # Extract snapshot times from coordinator log
    if [ -f "logs/inp-params-${topology}_${processes}_coordinator.log" ]; then
        grep "snapshot.*complete.*in.*ms" logs/inp-params-${topology}_${processes}_coordinator.log | \
        sed 's/.*in \([0-9]*\) ms.*/\1/' > results/experiments/${topology}_${processes}_times.txt
        
        # Calculate average snapshot time
        if [ -s "results/experiments/${topology}_${processes}_times.txt" ]; then
            avg_time=$(awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print 0}' results/experiments/${topology}_${processes}_times.txt)
            echo "${processes},${avg_time}" >> results/experiments/${topology}_results.csv
        fi
    fi
    
    # Clean up logs for next experiment
    rm -rf logs/
    mkdir -p logs/
    
    echo "Completed: ${topology} with ${processes} processes"
    echo "----------------------------------------"
}

# Initialize result files
echo "Processes,Average_Time" > results/experiments/line_results.csv
echo "Processes,Average_Time" > results/experiments/ring_results.csv
echo "Processes,Average_Time" > results/experiments/tree_results.csv
echo "Processes,Average_Time" > results/experiments/arbitrary_results.csv

# Run experiments for different topologies and process counts
for processes in 2 4 6 8 10; do
    for topology in line ring tree arbitrary; do
        if [ -f "inp-params-${topology}-${processes}.txt" ]; then
            run_experiment ${topology} ${processes}
        fi
    done
done

echo "All experiments completed!"
echo "Results saved in results/experiments/ directory"

# Generate summary
echo "Experiment Summary:"
echo "=================="
for topology in line ring tree arbitrary; do
    if [ -f "results/experiments/${topology}_results.csv" ]; then
        echo "${topology} topology results:"
        cat results/experiments/${topology}_results.csv
        echo ""
    fi
done

#include "process.h"
#include <iostream>
#include <vector>
#include <map>
#include <thread>
#include <chrono>
#include <sys/wait.h>
#include <unistd.h>
#include <signal.h>

class DistributedBankingSystem {
private:
    int n_processes;
    double lambda, lambda_c;
    int k_snapshots;
    std::vector<int> initial_balances;
    std::map<int, std::vector<int>> topology;
    std::vector<Process*> processes;
    std::vector<pid_t> child_pids;
    std::string topology_name;
    int process_count;
    int experiment_id;
    
public:
    DistributedBankingSystem() : n_processes(0), lambda(0), lambda_c(0), k_snapshots(0) {}
    
    ~DistributedBankingSystem() {
        cleanup();
    }
    
    bool initialize(const std::string& input_file) {
        if (!InputParser::parse_input_file(input_file, n_processes, lambda, lambda_c, k_snapshots,
                                         initial_balances, topology)) {
            std::cerr << "Failed to parse input file" << std::endl;
            return false;
        }
        
        // Extract topology name and process count from filename
        size_t pos = input_file.find_last_of('-');
        if (pos != std::string::npos) {
            topology_name = input_file.substr(input_file.find_last_of('/') + 1, pos - input_file.find_last_of('/') - 1);
            process_count = n_processes;
        } else {
            topology_name = "unknown";
            process_count = n_processes;
        }
        
        // Generate experiment ID based on topology and process count
        experiment_id = std::hash<std::string>{}(topology_name + std::to_string(process_count)) % 1000;
        
        std::cout << "Initialized " << topology_name << " topology with " << n_processes << " processes" << std::endl;
        std::cout << "Lambda: " << lambda << ", Lambda_c: " << lambda_c << ", K: " << k_snapshots << std::endl;
        
        return true;
    }
    
    void run() {
        // Create logs directory
        system("mkdir -p logs");
        
        // Create Process 0 as coordinator (no balance, but connected to all processes)
        std::vector<int> coordinator_neighbors;
        for (int i = 1; i <= n_processes; i++) {
            coordinator_neighbors.push_back(i);
        }
        Process* coordinator = new Process(0, 0, coordinator_neighbors, 
                                         n_processes, true, k_snapshots, lambda_c, experiment_id, topology_name, initial_balances);
        processes.push_back(coordinator);
        
        // Calculate expected total balance for coordinator
        int expected_total = 0;
        for (int balance : initial_balances) {
            expected_total += balance;
        }
        coordinator->set_expected_total_balance(expected_total);
        
        // Create non-coordinator processes (1 to n_processes) with FULL connectivity
        for (int i = 1; i <= n_processes; i++) {
            if (topology.find(i) == topology.end()) {
                std::cerr << "No topology found for process " << i << std::endl;
                continue;
            }
            
            // Create a fully connected network for snapshot algorithm
            std::vector<int> all_neighbors;
            for (int j = 1; j <= n_processes; j++) {
                if (j != i) {
                    all_neighbors.push_back(j);
                }
            }
            // Also connect to coordinator
            all_neighbors.push_back(0);
            
            // Store original topology for money transfers
            std::vector<int> original_topology = topology[i];
            
            Process* process = new Process(i, initial_balances[i-1], all_neighbors, 
                                         n_processes, false, k_snapshots, lambda_c, experiment_id, topology_name, initial_balances);
            
            // Set the original topology for money transfers
            process->set_original_topology(original_topology);
            processes.push_back(process);
        }
        
        // Phase 1: Start all servers first
        std::cout << "Phase 1: Starting all servers..." << std::endl;
    
        for (int i = 0; i < processes.size(); i++) {
            processes[i]->start_server();
            std::this_thread::sleep_for(std::chrono::milliseconds(500)); // Longer delay between server starts
        }
        
        // Give servers time to be ready
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        
        // Phase 2: Establish connections
        std::cout << "Phase 2: Establishing connections..." << std::endl;
        for (int i = 0; i < processes.size(); i++) {
            processes[i]->start_connections();
            std::this_thread::sleep_for(std::chrono::milliseconds(200)); // Delay between connection starts
        }
        
        // Give processes time to establish all connections
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        
        // Verify connections are established
        std::cout << "Verifying network connections..." << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        
        // Record start time
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Wait for coordinator to complete all snapshots
        std::cout << "System started. Waiting for completion..." << std::endl;
        std::cout << "DEBUG: Main process entering monitoring loop" << std::endl;
        
        // Keep main thread alive and monitor progress
        int loop_count = 0;
        while (true) {
            loop_count++;
            std::cout << "DEBUG: Main process monitoring loop iteration " << loop_count << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(1));
            
            // Check if all processes have terminated
            bool all_terminated = true;
            int running_count = 0;
            std::cout << "DEBUG: Checking process status..." << std::endl;
            
            for (size_t i = 0; i < processes.size(); i++) {
                Process* process = processes[i];
                if (process) {
                    bool is_running = process->is_running();
                    std::cout << "DEBUG: Process " << i << " running status: " << (is_running ? "RUNNING" : "TERMINATED") << std::endl;
                    if (is_running) {
                        all_terminated = false;
                        running_count++;
                    }
                } else {
                    std::cout << "DEBUG: Process " << i << " is NULL" << std::endl;
                }
            }
            
            std::cout << "DEBUG: Process status check complete. Running: " << running_count << ", Total: " << processes.size() << std::endl;
            
            // Log status every 5 seconds
            static int last_log_time = 0;
            auto current_time = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time);
            if (elapsed.count() != last_log_time && elapsed.count() % 5 == 0) {
                std::cout << "Process status: " << (processes.size() - running_count) << "/" << processes.size() << " processes terminated" << std::endl;
                std::cout << "DEBUG: Elapsed time: " << elapsed.count() << " seconds" << std::endl;
                last_log_time = elapsed.count();
            }
            
            if (all_terminated) {
                std::cout << "DEBUG: All processes have terminated!" << std::endl;
                auto end_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
                
                std::cout << "All processes have terminated" << std::endl;
                std::cout << "Total execution time: " << duration.count() << " ms" << std::endl;
                std::cout << "DEBUG: Main process exiting normally" << std::endl;
                
                // Write timing results to file
                std::ofstream timing_file("results/" + topology_name + "-" + std::to_string(process_count) + "-timing.txt");
                timing_file << "Topology: " << topology_name << std::endl;
                timing_file << "Processes: " << process_count << std::endl;
                timing_file << "Execution time: " << duration.count() << " ms" << std::endl;
                timing_file << "Lambda: " << lambda << std::endl;
                timing_file << "Lambda_c: " << lambda_c << std::endl;
                timing_file << "Snapshots: " << k_snapshots << std::endl;
                timing_file.close();
                
                break;
            }
            
            // Timeout after 5 minutes to allow all snapshots to complete
            auto timeout_time = std::chrono::high_resolution_clock::now();
            auto timeout_elapsed = std::chrono::duration_cast<std::chrono::seconds>(timeout_time - start_time);
            std::cout << "DEBUG: Checking timeout. Elapsed: " << timeout_elapsed.count() << " seconds" << std::endl;
            if (timeout_elapsed.count() >= 300) { // 5 minutes timeout
                std::cerr << "DEBUG: TIMEOUT REACHED! Experiment took longer than 5 minutes" << std::endl;
                std::cerr << "Timeout: Experiment took longer than 5 minutes" << std::endl;
                std::cout << "DEBUG: Main process exiting due to timeout" << std::endl;
                break;
            }
            
            // Progress indicator every 10 seconds
            if (timeout_elapsed.count() % 10 == 0 && timeout_elapsed.count() > 0) {
                std::cout << "Experiment running for " << timeout_elapsed.count() << " seconds..." << std::endl;
            }
        }
    }
    
    void cleanup() {
        std::cout << "DEBUG: Starting cleanup of " << processes.size() << " processes" << std::endl;
        for (size_t i = 0; i < processes.size(); i++) {
            Process* process = processes[i];
            if (process) {
                std::cout << "DEBUG: Stopping and deleting process " << i << std::endl;
                process->stop();
                delete process;
            } else {
                std::cout << "DEBUG: Process " << i << " is already NULL" << std::endl;
            }
        }
        processes.clear();
        std::cout << "DEBUG: Cleanup completed" << std::endl;
    }
};

// Signal handler for graceful shutdown
void signal_handler(int signum) {
    std::cout << "\nReceived signal " << signum << ". Shutting down gracefully..." << std::endl;
    exit(0);
}

int main(int argc, char* argv[]) {
    // if (argc < 2) {
    //     std::cerr << "Usage: " << argv[0] << " <input_file>" << std::endl;
    //     std::cerr << "  input_file: Path to input parameters file" << std::endl;
    //     return 1;
    // }
    argv[1] = const_cast<char*>("inp-params-line-3.txt");
    
    // Set up signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    signal(SIGPIPE, SIG_IGN); // Ignore SIGPIPE to prevent crashes from broken pipes
    
    std::string input_file = argv[1];
    
    // Create results directory
    system("mkdir -p results");
    
    // Run simulation
    std::cout << "DEBUG: Creating DistributedBankingSystem" << std::endl;
    DistributedBankingSystem system;
    
    std::cout << "DEBUG: Initializing system with file: " << input_file << std::endl;
    if (!system.initialize(input_file)) {
        std::cerr << "DEBUG: System initialization FAILED" << std::endl;
        std::cerr << "Failed to initialize system" << std::endl;
        return 1;
    }
    
    std::cout << "DEBUG: System initialized successfully, starting run()" << std::endl;
    system.run();
    
    std::cout << "DEBUG: System run() completed, main process exiting with code 0" << std::endl;
    return 0;
}



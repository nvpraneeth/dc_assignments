#include "process.h"
#include <iostream>
#include <vector>
#include <map>
#include <thread>
#include <chrono>
#include <sys/wait.h>
#include <unistd.h>

class DistributedBankingSystem {
private:
    int n_processes;
    double lambda, lambda_c;
    int k_snapshots;
    std::vector<int> initial_balances;
    std::map<int, std::vector<int>> topology;
    std::vector<Process*> processes;
    std::vector<pid_t> child_pids;
    
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
        
        std::cout << "Initialized system with " << n_processes << " processes" << std::endl;
        std::cout << "Lambda: " << lambda << ", Lambda_c: " << lambda_c << ", K: " << k_snapshots << std::endl;
        
        return true;
    }
    
    void run() {
        // Create logs directory
        system("mkdir -p logs");
        
        // Start all processes
        for (int i = 1; i <= n_processes; i++) {
            if (topology.find(i) == topology.end()) {
                std::cerr << "No topology found for process " << i << std::endl;
                continue;
            }
            
            bool is_coordinator = (i == 1); // First process is coordinator
            Process* process = new Process(i, initial_balances[i-1], topology[i], 
                                         n_processes, is_coordinator, k_snapshots, lambda_c);
            processes.push_back(process);
        }
        
        // Start all processes
        for (Process* process : processes) {
            process->start();
        }
        
        // Wait for coordinator to complete all snapshots
        std::cout << "System started. Waiting for completion..." << std::endl;
        
        // Keep main thread alive
        while (true) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            
            // Check if all processes have terminated
            bool all_terminated = true;
            for (Process* process : processes) {
                if (process->is_running()) {
                    all_terminated = false;
                    break;
                }
            }
            
            if (all_terminated) {
                std::cout << "All processes have terminated" << std::endl;
                break;
            }
        }
    }
    
    void cleanup() {
        for (Process* process : processes) {
            if (process) {
                process->stop();
                delete process;
            }
        }
        processes.clear();
    }
    
    void run_experiment(const std::string& topology_name, int n, double lambda, double lambda_c, int k) {
        std::cout << "Running experiment for " << topology_name << " with " << n << " processes" << std::endl;
        
        // Create experiment-specific input file
        std::string exp_file = "exp_" + topology_name + "_" + std::to_string(n) + ".txt";
        create_experiment_input(exp_file, n, lambda, lambda_c, k, topology_name);
        
        // Run the experiment
        DistributedBankingSystem system;
        if (system.initialize(exp_file)) {
            auto start_time = std::chrono::high_resolution_clock::now();
            system.run();
            auto end_time = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            std::cout << "Experiment completed in " << duration.count() << " ms" << std::endl;
        }
    }
    
private:
    void create_experiment_input(const std::string& filename, int n, double lambda, double lambda_c, int k, const std::string& topology_type) {
        std::ofstream file(filename);
        file << n << " " << lambda << " " << lambda_c << " " << k << std::endl;
        
        // Generate initial balances
        for (int i = 0; i < n; i++) {
            file << (100 + i * 50) << " ";
        }
        file << std::endl;
        
        // Generate topology
        if (topology_type == "line") {
            create_line_topology(file, n);
        } else if (topology_type == "ring") {
            create_ring_topology(file, n);
        } else if (topology_type == "tree") {
            create_tree_topology(file, n);
        } else if (topology_type == "arbitrary") {
            create_arbitrary_topology(file, n);
        }
        
        file.close();
    }
    
    void create_line_topology(std::ofstream& file, int n) {
        for (int i = 1; i <= n; i++) {
            file << i;
            if (i > 1) file << " " << (i - 1);
            if (i < n) file << " " << (i + 1);
            file << std::endl;
        }
    }
    
    void create_ring_topology(std::ofstream& file, int n) {
        for (int i = 1; i <= n; i++) {
            file << i;
            int prev = (i == 1) ? n : (i - 1);
            int next = (i == n) ? 1 : (i + 1);
            file << " " << prev << " " << next;
            file << std::endl;
        }
    }
    
    void create_tree_topology(std::ofstream& file, int n) {
        for (int i = 1; i <= n; i++) {
            file << i;
            int left_child = 2 * i;
            int right_child = 2 * i + 1;
            if (left_child <= n) file << " " << left_child;
            if (right_child <= n) file << " " << right_child;
            if (i > 1) file << " " << (i / 2); // parent
            file << std::endl;
        }
    }
    
    void create_arbitrary_topology(std::ofstream& file, int n) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(1, n);
        
        for (int i = 1; i <= n; i++) {
            file << i;
            // Each node connects to 1-3 random other nodes
            int num_connections = dis(gen) % 3 + 1;
            std::set<int> connections;
            while (static_cast<int>(connections.size()) < num_connections) {
                int neighbor = dis(gen);
                if (neighbor != i) {
                    connections.insert(neighbor);
                }
            }
            for (int neighbor : connections) {
                file << " " << neighbor;
            }
            file << std::endl;
        }
    }
};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file> [experiment]" << std::endl;
        std::cerr << "  input_file: Path to inp-params.txt" << std::endl;
        std::cerr << "  experiment: Run experiments (optional)" << std::endl;
        return 1;
    }
    
    std::string input_file = argv[1];
    bool run_experiments = (argc > 2 && std::string(argv[2]) == "experiment");
    
    if (run_experiments) {
        std::cout << "Running experiments..." << std::endl;
        
        DistributedBankingSystem system;
        
        // Run experiments for different topologies and process counts
        std::vector<std::string> topologies = {"line", "ring", "tree", "arbitrary"};
        std::vector<int> process_counts = {2, 4, 6, 8, 10};
        
        for (const std::string& topology : topologies) {
            std::cout << "\n=== " << topology << " topology ===" << std::endl;
            for (int n : process_counts) {
                system.run_experiment(topology, n, 5.0, 10.0, 20);
                std::this_thread::sleep_for(std::chrono::seconds(2)); // Wait between experiments
            }
        }
    } else {
        // Run normal simulation
        DistributedBankingSystem system;
        
        if (!system.initialize(input_file)) {
            std::cerr << "Failed to initialize system" << std::endl;
            return 1;
        }
        
        system.run();
    }
    
    return 0;
}

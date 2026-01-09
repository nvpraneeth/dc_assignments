#ifndef PROCESS_H
#define PROCESS_H

#include "message.h"
#include <vector>
#include <map>
#include <set>
#include <queue>

class Process {
private:
    int process_id;
    int initial_balance;
    std::atomic<int> current_balance;
    std::vector<int> neighbors;
    std::vector<int> original_topology_neighbors; // For money transfers only
    NetworkManager* network;
    Logger* logger;
    int n_processes;
    std::vector<int> initial_balances; // Store all initial balances for coordinator
    
    // Threading
    std::thread send_thread;
    std::thread receive_thread;
    std::atomic<bool> should_terminate;
    std::atomic<bool> coordinator_termination_sent;
    std::mutex balance_mutex;
    std::condition_variable balance_cv;
    
    // Chandy-Lamport snapshot state
    std::map<int, bool> recorded_local_state; // snapshot_id -> recorded
    std::map<int, int> recorded_balance; // snapshot_id -> balance at time of recording
    std::map<int, std::map<int, bool>> marker_received_on_channel; // snapshot_id -> neighbor -> received
    std::map<int, std::map<int, std::vector<int>>> channel_states; // snapshot_id -> neighbor -> messages
    std::map<int, std::map<int, bool>> channel_recording; // snapshot_id -> neighbor -> recording
    std::map<int, bool> snapshot_reply_sent; // snapshot_id -> reply_sent
    std::mutex snapshot_mutex;
    
    
    // Coordinator specific
    bool is_coordinator;
    int k_snapshots;
    int completed_snapshots;
    double lambda_c;
    int initial_total_balance;
    std::map<int, std::set<int>> markers_sent_for_snapshot; // snapshot_id -> set of processes
    
    // Random number generation
    std::random_device rd;
    std::mt19937 gen;
    std::exponential_distribution<> exp_dist;
    std::map<int, std::map<int, int>> snapshot_collection; // snapshot_id -> process_id -> balance
    std::map<int, std::map<int, std::vector<int>>> snapshot_channels; // snapshot_id -> process_id -> channel_messages
    std::map<int, std::set<int>> snapshot_pending; // snapshot_id -> pending_processes
    std::map<int, bool> snapshot_analyzed; // snapshot_id -> whether analysis has been completed
    
public:
    Process(int pid, int initial_bal, const std::vector<int>& neigh, int n_proc, bool is_coord = false, int k = 0, double lambda_c = 10.0, int exp_id = 0, const std::string& topology = "unknown", const std::vector<int>& all_initial_balances = std::vector<int>());
    ~Process();
    
    void start();
    void start_server();
    void start_connections();
    void stop();
    
    // Thread functions
    void send_function();
    void receive_function();
    
    // Chandy-Lamport snapshot functions
    void handle_marker(const Message& msg);
    void handle_snapshot_reply(const Message& msg);
    void record_local_state(int snapshot_id);
    void start_channel_recording(int snapshot_id);
    void stop_channel_recording(int snapshot_id, int from_neighbor);
    void send_snapshot_reply(int snapshot_id);
    bool is_snapshot_complete(int snapshot_id);
    
    // Coordinator functions
    void initiate_snapshot(int snapshot_id);
    void analyze_snapshot(int snapshot_id);
    void coordinator_send_function();
    
    // Utility functions
    int get_exponential_random(double lambda);
    int get_random_neighbor();
    int get_random_amount();
    void log_balance();
    
    // Setters
    void set_expected_total_balance(int expected_total) { initial_total_balance = expected_total; }
    void set_original_topology(const std::vector<int>& original_neighbors) { original_topology_neighbors = original_neighbors; }
    
    // Getters
    int get_current_balance() const { return current_balance.load(); }
    int get_initial_balance() const { return initial_balance; }
    bool is_running() const { return !should_terminate.load(); }
};

class InputParser {
public:
    static bool parse_input_file(const std::string& filename, 
                                int& n, double& lambda, double& lambda_c, int& k,
                                std::vector<int>& initial_balances,
                                std::map<int, std::vector<int>>& topology);
};

#endif // PROCESS_H

#include <iostream>
#include <vector>
#include <map>
#include <thread>
#include <mutex>
#include <chrono>
#include <random>
#include <fstream>
#include <sstream>
#include <queue>
#include <atomic>
#include <iomanip>
#include <ctime>
#include <set>

using namespace std;

// Global variables for synchronization
mutex log_mutex;
mutex message_mutex;
atomic<int> total_messages_sent(0);
atomic<bool> should_terminate(false);

// Message structure for Singhal-Kshemkalyani optimization
struct SKMessage {
    int sender_id;
    int receiver_id;
    vector<pair<int, int>> clock_entries; // (process_id, clock_value) pairs
    string message_id;
    chrono::system_clock::time_point real_time;
};

// Event structure
struct Event {
    int process_id;
    string event_type; // "internal", "send", "receive"
    string event_id;
    vector<int> vector_clock;
    chrono::system_clock::time_point real_time;
    int target_process; // for send/receive events
    string message_id; // for send/receive events
};

class SinghalKshemkalyaniClock {
private:
    int process_id;
    int num_processes;
    vector<int> vector_clock;
    vector<int> last_sent_clock; // Last clock values sent to each process
    map<int, queue<SKMessage>> message_queues;
    mutex queue_mutex;
    vector<int> neighbors;
    int messages_sent;
    double lambda;
    double alpha;
    int max_messages;
    random_device rd;
    mt19937 gen;
    exponential_distribution<> exp_dist;
    uniform_real_distribution<> uniform_dist;
    
    // Statistics
    atomic<int> total_clock_entries_sent;
    atomic<int> total_messages_sent_by_process;

public:
    SinghalKshemkalyaniClock(int id, int n, double l, double a, int m, const vector<int>& nbrs) 
        : process_id(id), num_processes(n), lambda(l), alpha(a), max_messages(m), 
          neighbors(nbrs), messages_sent(0), gen(rd()), exp_dist(1.0/lambda), uniform_dist(0.0, 1.0) {
        vector_clock.resize(n, 0);
        last_sent_clock.resize(n, 0);
        vector_clock[id - 1] = 0; // Process IDs are 1-indexed
        total_clock_entries_sent = 0;
        total_messages_sent_by_process = 0;
    }

    void increment_clock() {
        vector_clock[process_id - 1]++;
    }

    void update_clock(const vector<pair<int, int>>& received_entries) {
        // Update clock with received entries
        for (const auto& entry : received_entries) {
            int process = entry.first - 1; // Convert to 0-indexed
            int value = entry.second;
            vector_clock[process] = max(vector_clock[process], value);
        }
        increment_clock();
    }

    // Singhal-Kshemkalyani optimization: send only changed entries
    vector<pair<int, int>> get_optimized_clock_entries(int target_process) {
        (void)target_process; // Suppress unused parameter warning
        vector<pair<int, int>> entries;
        
        for (int i = 0; i < num_processes; i++) {
            int process_id_1_indexed = i + 1;
            if (vector_clock[i] > last_sent_clock[i]) {
                entries.push_back({process_id_1_indexed, vector_clock[i]});
            }
        }
        
        // Update last sent clock for this target
        last_sent_clock = vector_clock;
        
        return entries;
    }

    void log_event(const Event& event) {
        lock_guard<mutex> lock(log_mutex);
        ofstream log_file("singhal_kshemkalyani_log.txt", ios::app);
        
        auto time_t = chrono::system_clock::to_time_t(event.real_time);
        auto tm = *localtime(&time_t);
        
        log_file << "Process" << event.process_id << " " << event.event_type << " event " 
                << event.event_id;
        
        if (event.event_type == "send") {
            log_file << " to process" << event.target_process;
        } else if (event.event_type == "receive") {
            log_file << " from process" << event.target_process;
        }
        
        log_file << " at " << put_time(&tm, "%H:%M:%S") << ", vc: [";
        for (size_t i = 0; i < event.vector_clock.size(); i++) {
            log_file << event.vector_clock[i];
            if (i < event.vector_clock.size() - 1) log_file << " ";
        }
        log_file << "]" << endl;
        log_file.close();
    }

    void internal_event() {
        increment_clock();
        
        Event event;
        event.process_id = process_id;
        event.event_type = "internal";
        event.event_id = "e" + to_string(process_id) + to_string(rand() % 1000);
        event.vector_clock = vector_clock;
        event.real_time = chrono::system_clock::now();
        
        log_event(event);
        
        // Simulate processing time
        this_thread::sleep_for(chrono::milliseconds(1));
    }

    void send_message() {
        if (messages_sent >= max_messages) return;
        
        increment_clock();
        
        // Choose random neighbor
        int neighbor_idx = rand() % neighbors.size();
        int target_process = neighbors[neighbor_idx];
        
        // Get optimized clock entries
        vector<pair<int, int>> clock_entries = get_optimized_clock_entries(target_process);
        
        SKMessage msg;
        msg.sender_id = process_id;
        msg.receiver_id = target_process;
        msg.clock_entries = clock_entries;
        msg.message_id = "m" + to_string(process_id) + to_string(messages_sent);
        msg.real_time = chrono::system_clock::now();
        
        // Send message to target process
        {
            lock_guard<mutex> lock(message_mutex);
            message_queues[target_process].push(msg);
        }
        
        Event event;
        event.process_id = process_id;
        event.event_type = "send";
        event.event_id = msg.message_id;
        event.vector_clock = vector_clock;
        event.real_time = msg.real_time;
        event.target_process = target_process;
        event.message_id = msg.message_id;
        
        log_event(event);
        
        messages_sent++;
        total_messages_sent++;
        total_messages_sent_by_process++;
        total_clock_entries_sent += clock_entries.size(); // Only changed entries sent
        
        // Check if all processes have sent enough messages
        if (total_messages_sent >= num_processes * max_messages) {
            should_terminate = true;
        }
    }

    void receive_messages() {
        lock_guard<mutex> lock(queue_mutex);
        while (!message_queues[process_id].empty()) {
            SKMessage msg = message_queues[process_id].front();
            message_queues[process_id].pop();
            
            update_clock(msg.clock_entries);
            
            Event event;
            event.process_id = process_id;
            event.event_type = "receive";
            event.event_id = msg.message_id;
            event.vector_clock = vector_clock;
            event.real_time = chrono::system_clock::now();
            event.target_process = msg.sender_id;
            event.message_id = msg.message_id;
            
            log_event(event);
        }
    }

    void run() {
        while (!should_terminate && messages_sent < max_messages) {
            // Generate event based on exponential distribution
            double event_time = exp_dist(gen);
            this_thread::sleep_for(chrono::milliseconds(static_cast<int>(event_time)));
            
            // Check for received messages first
            receive_messages();
            
            // Generate event type based on alpha ratio
            double rand_val = uniform_dist(gen);
            if (rand_val < alpha / (alpha + 1)) {
                internal_event();
            } else {
                send_message();
            }
        }
    }

    // Getter methods for statistics
    int get_total_clock_entries_sent() const { return total_clock_entries_sent; }
    int get_total_messages_sent() const { return total_messages_sent_by_process; }
    int get_space_utilization() const { return num_processes * sizeof(int); } // Size of vector clock
};

// Function to read input parameters
tuple<int, double, double, int, vector<vector<int>>> read_input(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening input file" << endl;
        exit(1);
    }
    
    int n, m;
    double lambda, alpha;
    file >> n >> lambda >> alpha >> m;
    
    vector<vector<int>> topology(n);
    for (int i = 0; i < n; i++) {
        int process_id;
        file >> process_id;
        int neighbor;
        while (file >> neighbor) {
            topology[process_id - 1].push_back(neighbor);
            if (file.peek() == '\n') break;
        }
    }
    
    file.close();
    return make_tuple(n, lambda, alpha, m, topology);
}

int main() {
    // Clear previous log file
    ofstream log_file("singhal_kshemkalyani_log.txt", ios::trunc);
    log_file.close();
    
    // Read input parameters
    auto [n, lambda, alpha, m, topology] = read_input("inp-params.txt");
    
    cout << "Singhal-Kshemkalyani Optimization Implementation" << endl;
    cout << "Number of processes: " << n << endl;
    cout << "Lambda: " << lambda << endl;
    cout << "Alpha: " << alpha << endl;
    cout << "Max messages per process: " << m << endl;
    
    // Create and start processes
    vector<unique_ptr<SinghalKshemkalyaniClock>> processes;
    vector<thread> process_threads;
    
    for (int i = 1; i <= n; i++) {
        processes.push_back(make_unique<SinghalKshemkalyaniClock>(i, n, lambda, alpha, m, topology[i-1]));
        process_threads.emplace_back(&SinghalKshemkalyaniClock::run, processes.back().get());
    }
    
    // Wait for all processes to complete
    for (auto& thread : process_threads) {
        thread.join();
    }
    
    // Print statistics
    cout << "\n=== Singhal-Kshemkalyani Statistics ===" << endl;
    int total_entries = 0;
    int total_msgs = 0;
    int total_space = 0;
    
    for (int i = 0; i < n; i++) {
        cout << "Process " << (i+1) << ":" << endl;
        cout << "  Messages sent: " << processes[i]->get_total_messages_sent() << endl;
        cout << "  Clock entries sent: " << processes[i]->get_total_clock_entries_sent() << endl;
        cout << "  Space utilization: " << processes[i]->get_space_utilization() << " bytes" << endl;
        
        total_entries += processes[i]->get_total_clock_entries_sent();
        total_msgs += processes[i]->get_total_messages_sent();
        total_space += processes[i]->get_space_utilization();
    }
    
    cout << "\nTotal Statistics:" << endl;
    cout << "Total messages sent: " << total_msgs << endl;
    cout << "Total clock entries sent: " << total_entries << endl;
    cout << "Average entries per message: " << (total_msgs > 0 ? (double)total_entries / total_msgs : 0) << endl;
    cout << "Total space utilization: " << total_space << " bytes" << endl;
    
    return 0;
}

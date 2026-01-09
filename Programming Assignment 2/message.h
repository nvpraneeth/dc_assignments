#ifndef MESSAGE_H
#define MESSAGE_H

#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <mutex>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <random>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <set>
#include <queue>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <algorithm>

enum MessageType {
    XFER = 0,
    MARKER = 1,
    SNAP_REPLY = 2,
    TERM = 3,
    ACK = 4,
    UNKNOWN = -1
};

struct Message {
    MessageType type;
    int from;
    int to;
    int snapshot_id;
    int value;
    long timestamp;
    
    Message() : type(XFER), from(0), to(0), snapshot_id(0), value(0), timestamp(0) {}
    
    Message(MessageType t, int f, int t_dest, int snap_id, int val) 
        : type(t), from(f), to(t_dest), snapshot_id(snap_id), value(val) {
        timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    }
    
    std::string serialize() const {
        std::ostringstream oss;
        oss << type << " " << from << " " << to << " " << snapshot_id << " " << value << " " << timestamp;
        return oss.str();
    }
    
    static Message deserialize(const std::string& data) {
        std::istringstream iss(data);
        Message msg;
        int type_int;
        iss >> type_int >> msg.from >> msg.to >> msg.snapshot_id >> msg.value >> msg.timestamp;
        msg.type = static_cast<MessageType>(type_int);
        return msg;
    }
};

class NetworkManager {
private:
    int process_id;
    int port_base;
    int experiment_id;
    std::map<int, int> neighbor_sockets; // neighbor_id -> socket_fd
    std::map<int, int> neighbor_ports;   // neighbor_id -> port
    int server_socket;
    std::atomic<bool> running;
    std::mutex socket_mutex;
    std::queue<Message> message_queue; // Queue for incoming messages
    std::mutex queue_mutex;
    std::thread receive_thread;
    std::function<void(const Message&)> message_processor_callback; // Callback for direct message processing
    
public:
    NetworkManager(int pid, int base_port = 8000, int exp_id = 0) 
        : process_id(pid), port_base(base_port), experiment_id(exp_id), running(true) {
        server_socket = -1;
    }
    
    ~NetworkManager() {
        cleanup();
    }
    
    bool initialize(const std::vector<int>& neighbors);
    bool initialize_server();
    bool initialize_connections(const std::vector<int>& neighbors);
    bool send_message(int to, const Message& msg);
    bool send_message_persistent(int to, const Message& msg);
    bool send_message_on_socket(int sock, int to, const Message& msg);
    Message receive_message();
    void cleanup();
    int get_server_port() const { return port_base + (experiment_id * 100) + process_id; }
    
    // New methods for persistent connections
    void start_message_receiver();
    void stop_message_receiver();
    bool has_message();
    Message get_next_message();
    void set_message_processor_callback(std::function<void(const Message&)> callback);
    int get_queue_size(); // Get current queue size
    void process_connection(int client_socket); // Process incoming connection
    bool reconnect_to_neighbor(int neighbor_id);
};

class Logger {
private:
    std::ofstream log_file;
    std::mutex log_mutex;
    int process_id;
    bool is_coordinator;
    std::string topology_name;
    int process_count;
    
public:
    Logger(int pid, bool is_coord = false, const std::string& topology = "unknown", int n_processes = 0) 
        : process_id(pid), is_coordinator(is_coord), topology_name(topology), process_count(n_processes) {
        std::string filename;
        if (is_coordinator) {
            filename = "logs/" + topology_name + "_" + std::to_string(process_count) + "_coordinator.log";
        } else {
            filename = "logs/" + topology_name + "_" + std::to_string(process_count) + "_process_" + std::to_string(pid) + ".log";
        }
        log_file.open(filename, std::ios::app);
        
        // Write topology header to prevent stale log reading
        log_file << "=== " << topology_name << " topology - " << process_count << " processes - Process " << pid 
                 << (is_coordinator ? " (COORDINATOR)" : " (NON-COORDINATOR)") 
                 << " ===" << std::endl;
        log_file << "Timestamp: " << get_timestamp() << std::endl;
        log_file << "===========================================" << std::endl;
    }
    
    ~Logger() {
        if (log_file.is_open()) {
            log_file.close();
        }
    }
    
    void log_send(const Message& msg, const std::string& extra = "");
    void log_receive(const Message& msg, const std::string& extra = "");
    void log_snapshot(int snapshot_id, int balance, const std::string& action);
    void log_general(const std::string& message);
    
    // Coordinator-specific logging methods
    void log_snapshot_initiation(int snapshot_id, const std::string& target_processes);
    void log_snapshot_completion(int snapshot_id, int total_balance, int process_balances, int channel_balances);
    void log_snapshot_analysis(int snapshot_id, bool is_consistent, int expected_total, int actual_total);
    
    std::string get_timestamp() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
        
        std::ostringstream oss;
        oss << std::put_time(std::localtime(&time_t), "%H:%M:%S");
        oss << "." << std::setfill('0') << std::setw(3) << ms.count();
        return oss.str();
    }
};

#endif // MESSAGE_H

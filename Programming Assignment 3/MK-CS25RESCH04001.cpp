#include "common.h"

class MaekawaNode {
private:
    Config config;
    int port_base;
    std::vector<int> quorum;  // Processes in this node's quorum
    std::vector<std::thread> receiver_threads;
    std::ofstream logfile;
    
    // Socket management
    int server_socket;
    std::vector<int> client_sockets;
    std::vector<int> ports;
    
    // Algorithm state
    std::mutex state_mutex;
    std::vector<bool> locks_received;  // Whether we've received lock from quorum members
    std::queue<int> request_queue;  // Queue of pending requests
    bool requesting_cs;
    bool in_cs;
    int current_cs_number;
    
    // Statistics
    int total_messages_sent;
    double total_wait_time;
    
    // Coordinator statistics (for process 0)
    std::mutex stats_mutex;
    std::vector<int> process_messages;
    std::vector<double> process_wait_times;
    int stats_received_count;
    
    // Initialize quorum based on grid
    void init_grid_quorum() {
        int grid_size = static_cast<int>(sqrt(config.n));
        int row = config.process_id / grid_size;
        int col = config.process_id % grid_size;
        
        // Add all processes in same row
        for (int c = 0; c < grid_size; c++) {
            int pid = row * grid_size + c;
            if (pid != config.process_id) {
                quorum.push_back(pid);
            }
        }
        
        // Add all processes in same column
        for (int r = 0; r < grid_size; r++) {
            int pid = r * grid_size + col;
            if (pid != config.process_id && 
                std::find(quorum.begin(), quorum.end(), pid) == quorum.end()) {
                quorum.push_back(pid);
            }
        }
        
        log_message(logfile, "p" + std::to_string(config.process_id) + 
                   " quorum initialized with " + std::to_string(quorum.size()) + " processes");
    }
    
    // Initialize server socket
    void init_server() {
        struct sockaddr_in server_addr;
        int port = port_base + config.process_id;
        
        server_socket = socket(AF_INET, SOCK_STREAM, 0);
        if (server_socket < 0) {
            std::cerr << "Error creating socket for process " << config.process_id << std::endl;
            exit(1);
        }
        
        int opt = 1;
        setsockopt(server_socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
        
        memset(&server_addr, 0, sizeof(server_addr));
        server_addr.sin_family = AF_INET;
        server_addr.sin_addr.s_addr = INADDR_ANY;
        server_addr.sin_port = htons(port);
        
        if (bind(server_socket, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
            std::cerr << "Error binding socket for process " << config.process_id << std::endl;
            exit(1);
        }
        
        if (listen(server_socket, config.n) < 0) {
            std::cerr << "Error listening on socket for process " << config.process_id << std::endl;
            exit(1);
        }
        
        log_message(logfile, "p" + std::to_string(config.process_id) + 
                   " server started on port " + std::to_string(port));
    }
    
    // Connect to other processes
    void connect_to_peers() {
        client_sockets.resize(config.n, -1);
        
        for (int i = 0; i < config.n; i++) {
            if (i == config.process_id) continue;
            
            int sock = socket(AF_INET, SOCK_STREAM, 0);
            if (sock < 0) {
                std::cerr << "Error creating client socket for process " << i << std::endl;
                continue;
            }
            
            struct sockaddr_in peer_addr;
            memset(&peer_addr, 0, sizeof(peer_addr));
            peer_addr.sin_family = AF_INET;
            peer_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
            peer_addr.sin_port = htons(port_base + i);
            
            // Retry connection
            int retries = 100;
            while (retries > 0 && connect(sock, (struct sockaddr*)&peer_addr, sizeof(peer_addr)) < 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                retries--;
            }
            
            if (retries > 0) {
                client_sockets[i] = sock;
                log_message(logfile, "p" + std::to_string(config.process_id) + 
                           " connected to p" + std::to_string(i));
            } else {
                std::cerr << "Failed to connect to process " << i << std::endl;
            }
        }
    }
    
    // Accept incoming connections
    void accept_connections() {
        std::vector<std::thread> accept_threads;
        
        for (int i = 0; i < config.n - 1; i++) {
            accept_threads.emplace_back([this]() {
                struct sockaddr_in client_addr;
                socklen_t client_len = sizeof(client_addr);
                
                int client_sock = accept(server_socket, (struct sockaddr*)&client_addr, &client_len);
                if (client_sock < 0) {
                    return;
                }
                
                // Receive process ID from client
                int peer_id;
                recv(client_sock, &peer_id, sizeof(int), 0);
                
                std::lock_guard<std::mutex> lock(state_mutex);
                // Store socket for this peer (we'll handle it in message receiver)
                // For simplicity, we'll handle messages on accepted socket
            });
        }
    }
    
    // Send message to a process
    void send_message(int to_id, const Message& msg) {
        if (to_id == config.process_id || to_id < 0 || to_id >= config.n) return;
        
        char buffer[1024];
        int size = serialize_message(buffer, msg);
        
        // Send process ID first if connecting
        if (client_sockets[to_id] < 0) {
            // Try to establish connection
            int sock = socket(AF_INET, SOCK_STREAM, 0);
            struct sockaddr_in peer_addr;
            memset(&peer_addr, 0, sizeof(peer_addr));
            peer_addr.sin_family = AF_INET;
            peer_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
            peer_addr.sin_port = htons(port_base + to_id);
            
            if (connect(sock, (struct sockaddr*)&peer_addr, sizeof(peer_addr)) >= 0) {
                send(sock, &config.process_id, sizeof(int), 0);
                client_sockets[to_id] = sock;
            } else {
                return;
            }
        }
        
        if (client_sockets[to_id] >= 0) {
            send(client_sockets[to_id], buffer, size, 0);
            total_messages_sent++;
        }
    }
    
    // Receive messages
    void receive_messages() {
        std::vector<int> connected_peers(config.n, -1);
        
        // Accept all connections
        for (int i = 0; i < config.n - 1; i++) {
            struct sockaddr_in client_addr;
            socklen_t client_len = sizeof(client_addr);
            int client_sock = accept(server_socket, (struct sockaddr*)&client_addr, &client_len);
            
            if (client_sock >= 0) {
                int peer_id;
                recv(client_sock, &peer_id, sizeof(int), 0);
                connected_peers[peer_id] = client_sock;
                
                // Start receiver thread for this peer
                receiver_threads.emplace_back([this, client_sock]() {
                    char buffer[1024];
                    while (true) {
                        int bytes = recv(client_sock, buffer, sizeof(buffer), 0);
                        if (bytes <= 0) break;
                        
                        Message msg;
                        deserialize_message(buffer, msg);
                        handle_message(msg);
                    }
                });
            }
        }
        
        // Also send our ID to peers we connect to
        for (int i = 0; i < config.n; i++) {
            if (i != config.process_id && client_sockets[i] >= 0) {
                send(client_sockets[i], &config.process_id, sizeof(int), 0);
                
                // Start receiver thread
                receiver_threads.emplace_back([this, i]() {
                    char buffer[1024];
                    while (true) {
                        int bytes = recv(client_sockets[i], buffer, sizeof(buffer), 0);
                        if (bytes <= 0) break;
                        
                        Message msg;
                        deserialize_message(buffer, msg);
                        handle_message(msg);
                    }
                });
            }
        }
    }
    
    // Handle incoming message
    void handle_message(const Message& msg) {
        std::lock_guard<std::mutex> lock(state_mutex);
        
        if (msg.type == REQUEST) {
            log_message(logfile, "p" + std::to_string(config.process_id) + 
                       " receives p" + std::to_string(msg.from_id) + 
                       "'s request to enter CS at " + get_current_time());
            
            // Check if we should grant lock
            bool should_reply = false;
            if (!in_cs && !requesting_cs) {
                should_reply = true;
            } else if (requesting_cs) {
                // Add to queue (simplified - in real Maekawa, we might need ordering)
                request_queue.push(msg.from_id);
                should_reply = false;
            }
            
            if (should_reply) {
                Message reply_msg;
                reply_msg.type = REPLY;
                reply_msg.from_id = config.process_id;
                send_message(msg.from_id, reply_msg);
                
                log_message(logfile, "p" + std::to_string(config.process_id) + 
                           " replies to p" + std::to_string(msg.from_id) + 
                           "'s request to enter CS at " + get_current_time());
            }
        } else if (msg.type == REPLY) {
            log_message(logfile, "p" + std::to_string(config.process_id) + 
                       " receives p" + std::to_string(msg.from_id) + 
                       "'s reply at " + get_current_time());
            
            // Mark that we received lock from this quorum member
            if (requesting_cs) {
                // Find if this is in our quorum
                for (size_t i = 0; i < quorum.size(); i++) {
                    if (quorum[i] == msg.from_id) {
                        locks_received[i] = true;
                        break;
                    }
                }
            }
        } else if (msg.type == RELEASE) {
            log_message(logfile, "p" + std::to_string(config.process_id) + 
                       " receives p" + std::to_string(msg.from_id) + 
                       "'s release at " + get_current_time());
            
            // Check if we can grant lock to next in queue
            if (!request_queue.empty() && !in_cs && !requesting_cs) {
                int next_id = request_queue.front();
                request_queue.pop();
                
                Message reply_msg;
                reply_msg.type = REPLY;
                reply_msg.from_id = config.process_id;
                send_message(next_id, reply_msg);
                
                log_message(logfile, "p" + std::to_string(config.process_id) + 
                           " replies to p" + std::to_string(next_id) + 
                           "'s request to enter CS at " + get_current_time());
            }
        } else if (msg.type == TERM) {
            // Handle termination message from other processes
            if (config.process_id == 0) {
                std::lock_guard<std::mutex> stats_lock(stats_mutex);
                process_messages[msg.from_id] = msg.msg_count;
                process_wait_times[msg.from_id] = msg.total_cs_wait_time;
                stats_received_count++;
                log_message(logfile, "p" + std::to_string(config.process_id) + 
                           " received stats from p" + std::to_string(msg.from_id) + 
                           " at " + get_current_time());
            }
        }
    }
    
    // Request CS
    void request_cs(int cs_num) {
        std::unique_lock<std::mutex> lock(state_mutex);
        
        requesting_cs = true;
        current_cs_number = cs_num;
        locks_received.clear();
        locks_received.resize(quorum.size(), false);
        
        auto start_time = std::chrono::steady_clock::now();
        
        log_message(logfile, "p" + std::to_string(config.process_id) + 
                   " requests to enter CS at " + get_current_time() + 
                   " for the " + std::to_string(cs_num) + ordinal_suffix(cs_num) + " time");
        
        // Send REQUEST to all quorum members
        Message req_msg;
        req_msg.type = REQUEST;
        req_msg.from_id = config.process_id;
        req_msg.cs_number = cs_num;
        
        lock.unlock();
        
        for (int q : quorum) {
            send_message(q, req_msg);
            log_message(logfile, "p" + std::to_string(config.process_id) + 
                       " sends request to p" + std::to_string(q) + " at " + get_current_time());
        }
        
        // Wait for replies from all quorum members
        lock.lock();
        while (true) {
            bool all_replied = true;
            for (bool received : locks_received) {
                if (!received) {
                    all_replied = false;
                    break;
                }
            }
            if (all_replied) break;
            lock.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            lock.lock();
        }
        
        auto end_time = std::chrono::steady_clock::now();
        auto wait_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time).count() / 1000.0;
        total_wait_time += wait_time;
        
        requesting_cs = false;
        in_cs = true;
        
        log_message(logfile, "p" + std::to_string(config.process_id) + 
                   " enters CS at " + get_current_time() + 
                   " for the " + std::to_string(cs_num) + ordinal_suffix(cs_num) + " time");
    }
    
    // Release CS
    void release_cs(int cs_num) {
        std::lock_guard<std::mutex> lock(state_mutex);
        
        in_cs = false;
        
        log_message(logfile, "p" + std::to_string(config.process_id) + 
                   " leaves CS at " + get_current_time() + 
                   " for the " + std::to_string(cs_num) + ordinal_suffix(cs_num) + " time");
        
        // Send RELEASE to all quorum members
        Message release_msg;
        release_msg.type = RELEASE;
        release_msg.from_id = config.process_id;
        release_msg.cs_number = cs_num;
        
        for (int q : quorum) {
            send_message(q, release_msg);
            log_message(logfile, "p" + std::to_string(config.process_id) + 
                       " sends release to p" + std::to_string(q) + " at " + get_current_time());
        }
    }
    
    std::string ordinal_suffix(int n) {
        if (n % 100 >= 11 && n % 100 <= 13) return "th";
        switch (n % 10) {
            case 1: return "st";
            case 2: return "nd";
            case 3: return "rd";
            default: return "th";
        }
    }
    
public:
    MaekawaNode(const Config& cfg, int port_base) 
        : config(cfg), port_base(port_base), requesting_cs(false), in_cs(false),
          total_messages_sent(0), total_wait_time(0.0), stats_received_count(0) {
        
        if (config.process_id == 0) {
            process_messages.resize(config.n, 0);
            process_wait_times.resize(config.n, 0.0);
        }
        
        std::string log_filename = "p" + std::to_string(config.process_id) + ".log";
        logfile.open(log_filename);
        
        init_grid_quorum();
        init_server();
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100 * config.process_id));
        connect_to_peers();
        
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        
        std::thread receiver(&MaekawaNode::receive_messages, this);
        receiver.detach();
        
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
    
    void execute() {
        for (int i = 0; i < config.k; i++) {
            // Local computation
            double out_cs_time = exponential_random(config.alpha);
            log_message(logfile, "p" + std::to_string(config.process_id) + 
                       " is doing local computation at " + get_current_time());
            std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(out_cs_time * 1000)));
            
            // Request CS
            request_cs(i + 1);
            
            // Inside CS
            double in_cs_time = exponential_random(config.beta);
            std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(in_cs_time * 1000)));
            
            // Release CS
            release_cs(i + 1);
        }
        
        // Send statistics to coordinator (process 0)
        if (config.process_id != 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            Message term_msg;
            term_msg.type = TERM;
            term_msg.from_id = config.process_id;
            term_msg.msg_count = total_messages_sent;
            term_msg.total_cs_wait_time = total_wait_time;
            send_message(0, term_msg);
            log_message(logfile, "p" + std::to_string(config.process_id) + 
                       " sent stats to coordinator at " + get_current_time());
        } else {
            // Coordinator collects statistics
            process_messages[0] = total_messages_sent;
            process_wait_times[0] = total_wait_time;
            collect_statistics();
        }
    }
    
    void collect_statistics() {
        // Wait for all processes to finish and send their stats
        int max_wait = 60;  // Wait up to 60 seconds
        int waited = 0;
        while (stats_received_count < config.n - 1 && waited < max_wait) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            waited++;
            std::lock_guard<std::mutex> lock(stats_mutex);
            if (stats_received_count >= config.n - 1) break;
        }
        
        // Calculate statistics
        int total_msgs = 0;
        double total_wait = 0.0;
        for (int i = 0; i < config.n; i++) {
            total_msgs += process_messages[i];
            total_wait += process_wait_times[i];
        }
        
        double avg_msg_per_cs = (double)total_msgs / (config.n * config.k);
        double avg_wait_time = total_wait / (config.n * config.k);
        
        std::ofstream stats_file("mk_stats.txt");
        stats_file << "Maekawa Algorithm Statistics" << std::endl;
        stats_file << "========================================" << std::endl;
        stats_file << "Number of processes (n): " << config.n << std::endl;
        stats_file << "CS entries per process (k): " << config.k << std::endl;
        stats_file << "Total messages sent by all processes: " << total_msgs << std::endl;
        stats_file << "Average messages per CS entry: " << avg_msg_per_cs << std::endl;
        stats_file << "Total wait time (all processes): " << total_wait << " seconds" << std::endl;
        stats_file << "Average wait time per CS entry: " << avg_wait_time << " seconds" << std::endl;
        stats_file << "========================================" << std::endl;
        stats_file << "Per-process breakdown:" << std::endl;
        for (int i = 0; i < config.n; i++) {
            stats_file << "  Process " << i << ": " << process_messages[i] 
                      << " messages, " << process_wait_times[i] / config.k 
                      << " avg wait time" << std::endl;
        }
        stats_file.close();
        
        log_message(logfile, "p" + std::to_string(config.process_id) + 
                   " collected statistics from all processes at " + get_current_time());
    }
    
    ~MaekawaNode() {
        logfile.close();
        close(server_socket);
        for (int sock : client_sockets) {
            if (sock >= 0) close(sock);
        }
    }
};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <process_id>" << std::endl;
        return 1;
    }
    
    Config config;
    std::ifstream infile("inp-params.txt");
    if (!infile) {
        std::cerr << "Error opening inp-params.txt" << std::endl;
        return 1;
    }
    
    infile >> config.n >> config.k >> config.alpha >> config.beta;
    infile.close();
    
    config.process_id = std::stoi(argv[1]);
    
    // Check if n is a perfect square
    int grid_size = static_cast<int>(sqrt(config.n));
    if (grid_size * grid_size != config.n) {
        std::cerr << "Error: n must be a perfect square for Maekawa grid quorum" << std::endl;
        return 1;
    }
    
    int port_base = 10000;
    MaekawaNode node(config, port_base);
    node.execute();
    
    std::this_thread::sleep_for(std::chrono::seconds(10));
    
    return 0;
}

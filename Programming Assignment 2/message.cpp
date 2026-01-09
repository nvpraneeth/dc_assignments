#include "message.h"
#include <iomanip>
#include <ctime>
#include <algorithm>
#include <fcntl.h>
#include <unistd.h>

bool NetworkManager::initialize_server() {
    std::cout << "DEBUG: Process " << process_id << " initializing server on port " << get_server_port() << std::endl;
    
    // Clean up any existing server socket
    if (server_socket >= 0) {
        std::cout << "DEBUG: Process " << process_id << " cleaning up existing server socket: " << server_socket << std::endl;
        close(server_socket);
        server_socket = -1;
    }
    
    // Create server socket
    server_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket < 0) {
        std::cerr << "Process " << process_id << ": Failed to create server socket (errno: " << errno << ")" << std::endl;
        return false;
    }
    
    std::cout << "DEBUG: Process " << process_id << " created server socket: " << server_socket << std::endl;
    
    // Set socket options
    int opt = 1;
    if (setsockopt(server_socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        std::cerr << "Process " << process_id << ": Failed to set socket options (errno: " << errno << ")" << std::endl;
        close(server_socket);
        server_socket = -1;
        return false;
    }
    
    std::cout << "DEBUG: Process " << process_id << " set socket options successfully" << std::endl;
    
    // Bind to port with retries
    struct sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(get_server_port());
    
    std::cout << "DEBUG: Process " << process_id << " attempting to bind to port " << get_server_port() << std::endl;
    
    // Try binding with a small delay to handle race conditions
    for (int attempt = 1; attempt <= 3; attempt++) {
        if (bind(server_socket, (struct sockaddr*)&server_addr, sizeof(server_addr)) >= 0) {
            std::cout << "DEBUG: Process " << process_id << " bound to port " << get_server_port() << " successfully on attempt " << attempt << std::endl;
            break;
        }
        
        std::cout << "DEBUG: Process " << process_id << " bind attempt " << attempt << " failed (errno: " << errno << ")" << std::endl;
        if (attempt < 3) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        } else {
            std::cerr << "Process " << process_id << ": Failed to bind to port " << get_server_port() << " after 3 attempts (errno: " << errno << ")" << std::endl;
            close(server_socket);
            server_socket = -1;
            return false;
        }
    }
    
    if (listen(server_socket, 10) < 0) {
        std::cerr << "Process " << process_id << ": Failed to listen (errno: " << errno << ")" << std::endl;
        close(server_socket);
        server_socket = -1;
        return false;
    }
    
    std::cout << "Process " << process_id << " listening on port " << get_server_port() << std::endl;
    std::cout << "DEBUG: Process " << process_id << " server initialization completed successfully" << std::endl;
    return true;
}

bool NetworkManager::initialize_connections(const std::vector<int>& neighbors) {
    std::cout << "DEBUG: Process " << process_id << " starting to connect to " << neighbors.size() << " neighbors" << std::endl;
    
    // Connect to neighbors - establish all required connections
    // Use delays and retries to handle connection timing issues
    for (int neighbor : neighbors) {
        if (neighbor != process_id) {
            int neighbor_port = port_base + (experiment_id * 100) + neighbor;
            std::cout << "DEBUG: Process " << process_id << " attempting to connect to Process " << neighbor << " on port " << neighbor_port << std::endl;
            
            int sock = socket(AF_INET, SOCK_STREAM, 0);
            if (sock < 0) {
                std::cerr << "DEBUG: Process " << process_id << ": Failed to create socket for neighbor " << neighbor << std::endl;
                std::cerr << "Process " << process_id << ": Failed to create socket for neighbor " << neighbor << std::endl;
                continue;
            }
            
            struct sockaddr_in neighbor_addr;
            neighbor_addr.sin_family = AF_INET;
            neighbor_addr.sin_port = htons(neighbor_port);
            neighbor_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
            
            // Try to connect with retries
            int retries = 20; // More retries
            std::cout << "DEBUG: Process " << process_id << " trying to connect to Process " << neighbor << " (attempt 1/20)" << std::endl;
            std::cout << "DEBUG: Process " << process_id << " neighbor_addr.sin_family=" << neighbor_addr.sin_family << ", sin_port=" << neighbor_addr.sin_port << ", sin_addr=" << neighbor_addr.sin_addr.s_addr << std::endl;
            while (retries > 0) {
                std::cout << "DEBUG: Process " << process_id << " about to call connect() for Process " << neighbor << std::endl;
                if (connect(sock, (struct sockaddr*)&neighbor_addr, sizeof(neighbor_addr)) >= 0) {
                    std::cout << "DEBUG: Process " << process_id << " successfully connected to Process " << neighbor << " on attempt " << (21 - retries) << std::endl;
                    break;
                }
                std::cout << "DEBUG: Process " << process_id << " connection attempt " << (21 - retries) << " failed, retrying..." << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(300)); // Longer delay between retries
                retries--;
            }
            
            if (retries > 0) {
                std::cout << "DEBUG: Process " << process_id << " about to store socket " << sock << " for Process " << neighbor << " in neighbor_sockets" << std::endl;
                std::cout << "DEBUG: Process " << process_id << " neighbor_sockets size before: " << neighbor_sockets.size() << std::endl;
                std::cout << "DEBUG: Process " << process_id << " neighbor_sockets address: " << &neighbor_sockets << std::endl;
                try {
                    neighbor_sockets[neighbor] = sock;
                    std::cout << "DEBUG: Process " << process_id << " stored socket for Process " << neighbor << " in neighbor_sockets" << std::endl;
                    std::cout << "DEBUG: Process " << process_id << " neighbor_sockets size after: " << neighbor_sockets.size() << std::endl;
                } catch (const std::exception& e) {
                    std::cerr << "DEBUG: Process " << process_id << " exception storing socket: " << e.what() << std::endl;
                    close(sock);
                    continue;
                } catch (...) {
                    std::cerr << "DEBUG: Process " << process_id << " unknown exception storing socket" << std::endl;
                    close(sock);
                    continue;
                }
                std::cout << "DEBUG: Process " << process_id << " about to print connection message for neighbor " << neighbor << std::endl;
                std::cout << "Process " << process_id << " connected to neighbor " << neighbor << std::endl;
                std::cout << "DEBUG: Process " << process_id << " printed connection message for neighbor " << neighbor << std::endl;
            } else {
                std::cerr << "DEBUG: Process " << process_id << ": Failed to connect to neighbor " << neighbor << " after 20 retries" << std::endl;
                std::cerr << "Process " << process_id << ": Failed to connect to neighbor " << neighbor << " after 20 retries" << std::endl;
                close(sock);
            }
            
            // Add a small delay between connection attempts to prevent deadlocks
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    
    std::cout << "DEBUG: Process " << process_id << " finished connecting. Total connections: " << neighbor_sockets.size() << std::endl;
    return true;
}

bool NetworkManager::initialize(const std::vector<int>& neighbors) {
    // First initialize server
    if (!initialize_server()) {
        return false;
    }
    
    // Then establish connections
    return initialize_connections(neighbors);
}

bool NetworkManager::send_message(int to, const Message& msg) {
    std::cout << "DEBUG: Process " << process_id << " attempting to send message to Process " << to << std::endl;
    
    // Special debugging for TERM messages
    if (msg.type == TERM) {
        std::cout << "DEBUG: *** TERM MESSAGE SENDING *** Process " << process_id << " attempting to send TERM message to Process " << to << std::endl;
        std::cout << "DEBUG: *** TERM MESSAGE SENDING *** Message details - type=" << static_cast<int>(msg.type) << ", from=" << msg.from << ", to=" << msg.to << ", value=" << msg.value << ", snapshot_id=" << msg.snapshot_id << std::endl;
    }
    
    // Check if we have a persistent connection and if it's still valid
    std::lock_guard<std::mutex> lock(socket_mutex);
    auto it = neighbor_sockets.find(to);
    if (it != neighbor_sockets.end()) {
        // Test if the connection is still valid by trying to send a small amount of data
        int sock = it->second;
        int error = 0;
        socklen_t len = sizeof(error);
        int retval = getsockopt(sock, SOL_SOCKET, SO_ERROR, &error, &len);
        
        if (retval == 0 && error == 0) {
            std::cout << "DEBUG: Process " << process_id << " using existing persistent connection " << sock << " to Process " << to << std::endl;
            bool success = send_message_on_socket(sock, to, msg);
            if (success) {
                return true;
            } else {
                // Connection is broken, remove it and create a new one
                std::cout << "DEBUG: Process " << process_id << " persistent connection " << sock << " is broken, removing and creating new connection" << std::endl;
                close(sock);
                neighbor_sockets.erase(it);
            }
        } else {
            // Connection is broken, remove it and create a new one
            std::cout << "DEBUG: Process " << process_id << " persistent connection " << sock << " is broken (error: " << error << "), removing and creating new connection" << std::endl;
            close(sock);
            neighbor_sockets.erase(it);
        }
    }
    
    // Create a new connection for this message
    int target_port = port_base + (experiment_id * 100) + to;
    std::cout << "DEBUG: Process " << process_id << " target port for Process " << to << ": " << target_port << std::endl;
    
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        std::cerr << "DEBUG: Process " << process_id << ": Failed to create socket for message to Process " << to << " (errno: " << errno << ")" << std::endl;
        return false;
    }
    
    std::cout << "DEBUG: Process " << process_id << " created socket " << sock << " for Process " << to << std::endl;
    
    // Set socket options for better reliability
    int opt = 1;
    if (setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        std::cerr << "DEBUG: Process " << process_id << ": Failed to set socket options (errno: " << errno << ")" << std::endl;
    }
    
    struct sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(target_port);
    server_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    
    std::cout << "DEBUG: Process " << process_id << " attempting to connect to Process " << to << " on port " << target_port << std::endl;
    
        // Connect with retries for better reliability
        int connect_attempts = 0;
        int max_attempts = 5;  // Increased retry attempts for better reliability
        
        while (connect_attempts < max_attempts) {
            connect_attempts++;
            std::cout << "DEBUG: Process " << process_id << " connection attempt " << connect_attempts << "/" << max_attempts << " to Process " << to << std::endl;
            
            int connect_result = connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr));
            if (connect_result == 0) {
                std::cout << "DEBUG: Process " << process_id << " successfully connected to Process " << to << " on attempt " << connect_attempts << std::endl;
                break;
            }
            
            std::cerr << "DEBUG: Process " << process_id << ": Failed to connect to Process " << to << " on attempt " << connect_attempts << " (errno: " << errno << ")" << std::endl;
            
            if (connect_attempts < max_attempts) {
                std::cout << "DEBUG: Process " << process_id << " waiting before retry..." << std::endl;
                // Longer delay to allow target process to be ready
                int delay_ms = 500 * connect_attempts;  // 500ms, 1000ms, 1500ms, 2000ms, 2500ms
                std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
            } else {
                std::cerr << "DEBUG: Process " << process_id << ": All connection attempts failed to Process " << to << std::endl;
                close(sock);
                return false;
            }
        }
    
    // Send the message using the new socket
    bool success = send_message_on_socket(sock, to, msg);
    
    if (success) {
        std::cout << "DEBUG: Process " << process_id << " successfully sent complete message to Process " << to << std::endl;
        
        // Store this connection for future use instead of closing it
        neighbor_sockets[to] = sock;
        std::cout << "DEBUG: Process " << process_id << " stored persistent connection " << sock << " for Process " << to << std::endl;
        
        // Give receiver more time to fully process the message
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        
        std::cout << "DEBUG: Process " << process_id << " successfully sent complete message to Process " << to << " and kept connection alive" << std::endl;
    } else {
        std::cerr << "DEBUG: Process " << process_id << " failed to send message to Process " << to << std::endl;
        close(sock);
        return false;
    }
    
    // Special debugging for successful TERM message sends
    if (msg.type == TERM) {
        std::cout << "DEBUG: *** TERM MESSAGE SENT SUCCESSFULLY *** Process " << process_id << " successfully sent TERM message to Process " << to << std::endl;
    }
    
    return true;
}

bool NetworkManager::send_message_on_socket(int sock, int to, const Message& msg) {
    std::string data = msg.serialize();
    size_t len = data.length();
    
    std::cout << "DEBUG: Process " << process_id << " sending message on socket " << sock << " to Process " << to << " (message length: " << len << ")" << std::endl;
    
    // Send length first with MSG_NOSIGNAL to prevent SIGPIPE
    ssize_t bytes_sent = send(sock, &len, sizeof(len), MSG_NOSIGNAL);
    if (bytes_sent < 0) {
        std::cerr << "DEBUG: Process " << process_id << ": Failed to send message length on socket " << sock << " to " << to << " (errno: " << errno << ")" << std::endl;
        return false;
    }
    
    if (static_cast<size_t>(bytes_sent) != sizeof(len)) {
        std::cerr << "DEBUG: Process " << process_id << ": Partial send of message length on socket " << sock << " to " << to << " (" << bytes_sent << "/" << sizeof(len) << " bytes)" << std::endl;
        return false;
    }
    
    std::cout << "DEBUG: Process " << process_id << " successfully sent message length on socket " << sock << " to Process " << to << std::endl;
    
    // Send data with MSG_NOSIGNAL to prevent SIGPIPE
    bytes_sent = send(sock, data.c_str(), len, MSG_NOSIGNAL);
    if (bytes_sent < 0) {
        std::cerr << "DEBUG: Process " << process_id << ": Failed to send message data on socket " << sock << " to " << to << " (errno: " << errno << ")" << std::endl;
        return false;
    }
    
    if (static_cast<size_t>(bytes_sent) != len) {
        std::cerr << "DEBUG: Process " << process_id << ": Partial send of message data on socket " << sock << " to " << to << " (" << bytes_sent << "/" << len << " bytes)" << std::endl;
        return false;
    }
    
    std::cout << "DEBUG: Process " << process_id << " successfully sent message data on socket " << sock << " to Process " << to << std::endl;
    
    return true;
}

bool NetworkManager::send_message_persistent(int to, const Message& msg) {
    std::cout << "DEBUG: Process " << process_id << " attempting to send message via persistent connection to Process " << to << std::endl;
    
    // Special debugging for TERM messages
    if (msg.type == TERM) {
        std::cout << "DEBUG: *** TERM MESSAGE SENDING *** Process " << process_id << " attempting to send TERM message via persistent connection to Process " << to << std::endl;
        std::cout << "DEBUG: *** TERM MESSAGE SENDING *** Message details - type=" << static_cast<int>(msg.type) << ", from=" << msg.from << ", to=" << msg.to << ", value=" << msg.value << ", snapshot_id=" << msg.snapshot_id << std::endl;
    }
    
    std::lock_guard<std::mutex> lock(socket_mutex);
    
    // Check if we have a persistent connection to this neighbor
    auto it = neighbor_sockets.find(to);
    if (it == neighbor_sockets.end()) {
        std::cerr << "DEBUG: Process " << process_id << ": No persistent connection found to Process " << to << ", falling back to new connection" << std::endl;
        return send_message(to, msg);  // Fall back to regular send_message
    }
    
    int sock = it->second;
    std::cout << "DEBUG: Process " << process_id << " using persistent connection " << sock << " to Process " << to << std::endl;
    
    std::string data = msg.serialize();
    size_t len = data.length();
    
    std::cout << "DEBUG: Process " << process_id << " sending message via persistent connection to Process " << to << " (message length: " << len << ")" << std::endl;
    std::cout << "DEBUG: Process " << process_id << " message data: " << data << std::endl;
    
    // Send length first with MSG_NOSIGNAL to prevent SIGPIPE
    ssize_t bytes_sent = send(sock, &len, sizeof(len), MSG_NOSIGNAL);
    if (bytes_sent < 0) {
        std::cerr << "DEBUG: Process " << process_id << ": Failed to send message length via persistent connection to " << to << " (errno: " << errno << ")" << std::endl;
        std::cerr << "DEBUG: Process " << process_id << ": Connection may be broken, will fall back to new connection" << std::endl;
        return send_message(to, msg);  // Fall back to regular send_message
    }
    
    if (static_cast<size_t>(bytes_sent) != sizeof(len)) {
        std::cerr << "DEBUG: Process " << process_id << ": Partial send of message length via persistent connection to " << to << " (" << bytes_sent << "/" << sizeof(len) << " bytes)" << std::endl;
        return send_message(to, msg);  // Fall back to regular send_message
    }
    
    std::cout << "DEBUG: Process " << process_id << " successfully sent message length via persistent connection to Process " << to << std::endl;
    
    // Send data with MSG_NOSIGNAL to prevent SIGPIPE
    bytes_sent = send(sock, data.c_str(), len, MSG_NOSIGNAL);
    if (bytes_sent < 0) {
        std::cerr << "DEBUG: Process " << process_id << ": Failed to send message data via persistent connection to " << to << " (errno: " << errno << ")" << std::endl;
        return send_message(to, msg);  // Fall back to regular send_message
    }
    
    if (static_cast<size_t>(bytes_sent) != len) {
        std::cerr << "DEBUG: Process " << process_id << ": Partial send of message data via persistent connection to " << to << " (" << bytes_sent << "/" << len << " bytes)" << std::endl;
        return send_message(to, msg);  // Fall back to regular send_message
    }
    
    std::cout << "DEBUG: Process " << process_id << " successfully sent message via persistent connection to Process " << to << std::endl;
    
    // Keep the connection open for reuse (don't close it)
    return true;
}

Message NetworkManager::receive_message() {
    std::lock_guard<std::mutex> lock(queue_mutex);
    
    if (message_queue.empty()) {
        return Message(); // Return empty message if no messages
    }
    
    Message msg = message_queue.front();
    message_queue.pop();
    
    std::cout << "DEBUG: Process " << process_id << " retrieved message from queue: type=" << static_cast<int>(msg.type) << " from=" << msg.from << " to=" << msg.to << " value=" << msg.value << " (queue size: " << message_queue.size() << ")" << std::endl;
    
    // Special debugging for TERM messages being retrieved
    if (msg.type == TERM) {
        std::cout << "DEBUG: *** TERM MESSAGE RETRIEVED *** Process " << process_id << " retrieved TERM message from queue from Process " << msg.from << std::endl;
        std::cout << "DEBUG: *** TERM MESSAGE RETRIEVED *** Queue size after retrieval: " << message_queue.size() << std::endl;
    }
    
    return msg;
}

bool NetworkManager::reconnect_to_neighbor(int neighbor_id) {
    std::cout << "DEBUG: Process " << process_id << " attempting to reconnect to Process " << neighbor_id << std::endl;
    
    // Close the old socket if it exists
    if (neighbor_sockets.find(neighbor_id) != neighbor_sockets.end()) {
        close(neighbor_sockets[neighbor_id]);
        neighbor_sockets.erase(neighbor_id);
    }
    
    // Calculate the port for the neighbor
    int neighbor_port = port_base + (experiment_id * 100) + neighbor_id;
    
    // Create new socket
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        std::cerr << "DEBUG: Process " << process_id << " failed to create socket for reconnection to " << neighbor_id << std::endl;
        return false;
    }
    
    // Set up address structure
    struct sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(neighbor_port);
    server_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    
    // Attempt to connect with retries
    for (int attempt = 1; attempt <= 5; attempt++) {
        std::cout << "DEBUG: Process " << process_id << " trying to reconnect to Process " << neighbor_id << " (attempt " << attempt << "/5)" << std::endl;
        
        if (connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) == 0) {
            std::cout << "DEBUG: Process " << process_id << " successfully reconnected to Process " << neighbor_id << " on attempt " << attempt << std::endl;
            neighbor_sockets[neighbor_id] = sock;
            return true;
        }
        
        std::cout << "DEBUG: Process " << process_id << " reconnection attempt " << attempt << " failed, waiting before retry" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(100 * attempt));
    }
    
    std::cerr << "DEBUG: Process " << process_id << " failed to reconnect to Process " << neighbor_id << " after 5 attempts" << std::endl;
    close(sock);
    return false;
}

void NetworkManager::cleanup() {
    running = false;
    
    if (receive_thread.joinable()) {
        receive_thread.join();
    }
    
    for (auto& pair : neighbor_sockets) {
        close(pair.second);
    }
    neighbor_sockets.clear();
    
    if (server_socket >= 0) {
        close(server_socket);
        server_socket = -1;
    }
}

void NetworkManager::start_message_receiver() {
    std::cout << "DEBUG: Process " << process_id << " starting message receiver thread" << std::endl;
    std::cout << "DEBUG: Process " << process_id << " server socket: " << server_socket << std::endl;
    std::cout << "DEBUG: Process " << process_id << " running flag: " << running.load() << std::endl;
    
    // Validate server socket before starting thread
    if (server_socket < 0) {
        std::cerr << "ERROR: Process " << process_id << " cannot start message receiver - server socket is invalid (" << server_socket << ")" << std::endl;
        return;
    }
    
    // Test server socket is actually listening
    int socket_error = 0;
    socklen_t len = sizeof(socket_error);
    if (getsockopt(server_socket, SOL_SOCKET, SO_ERROR, &socket_error, &len) < 0) {
        std::cerr << "ERROR: Process " << process_id << " failed to get socket error status" << std::endl;
        return;
    }
    if (socket_error != 0) {
        std::cerr << "ERROR: Process " << process_id << " server socket has error: " << socket_error << std::endl;
        return;
    }
    
    std::cout << "DEBUG: Process " << process_id << " server socket validation passed" << std::endl;
    
    // Set server socket to non-blocking mode for better control
    int flags = fcntl(server_socket, F_GETFL, 0);
    if (flags >= 0) {
        fcntl(server_socket, F_SETFL, flags | O_NONBLOCK);
        std::cout << "DEBUG: Process " << process_id << " set server socket to non-blocking mode" << std::endl;
    }
    
    // Create thread with comprehensive error handling
    try {
        receive_thread = std::thread([this]() {
            std::cout << "DEBUG: Process " << process_id << " message receiver thread started successfully" << std::endl;
            std::cout << "DEBUG: Process " << process_id << " thread ID: " << std::this_thread::get_id() << std::endl;
            std::cout << "DEBUG: Process " << process_id << " server socket in thread: " << server_socket << std::endl;
            std::cout << "DEBUG: Process " << process_id << " running flag in thread: " << running.load() << std::endl;
            
            int connection_count = 0;
            int iteration_count = 0;
            
            while (running.load()) {
                try {
                    iteration_count++;
                    
                    // Validate server socket before each accept
                    if (server_socket < 0) {
                        std::cout << "ERROR: Process " << process_id << " server socket became invalid, stopping receiver thread" << std::endl;
                        break;
                    }
                    
                    // Log every 100 iterations to show the thread is alive
                    if (iteration_count % 100 == 0) {
                        std::cout << "DEBUG: Process " << process_id << " message receiver iteration " << iteration_count << ", waiting for connections..." << std::endl;
                    }
                    
                    // Use select() to check for incoming connections with timeout
                    fd_set read_fds;
                    FD_ZERO(&read_fds);
                    FD_SET(server_socket, &read_fds);
                    
                    struct timeval timeout;
                    timeout.tv_sec = 0;
                    timeout.tv_usec = 100000; // 100ms timeout
                    
                    int select_result = select(server_socket + 1, &read_fds, NULL, NULL, &timeout);
                    
                    if (select_result < 0) {
                        if (errno != EINTR) {
                            std::cout << "DEBUG: Process " << process_id << " select() failed (errno: " << errno << ")" << std::endl;
                        }
                        continue;
                    }
                    
                    if (select_result == 0) {
                        // Timeout - no connection available, continue
                        continue;
                    }
                    
                    if (!FD_ISSET(server_socket, &read_fds)) {
                        continue;
                    }
                    
                    std::cout << "DEBUG: Process " << process_id << " incoming connection detected on port " << get_server_port() << std::endl;
                    
                    // Accept incoming connection
                    struct sockaddr_in client_addr;
                    socklen_t client_len = sizeof(client_addr);
                    int client_socket = accept(server_socket, (struct sockaddr*)&client_addr, &client_len);
                    
                    std::cout << "DEBUG: Process " << process_id << " accept returned: " << client_socket << std::endl;
                    
                    if (client_socket < 0) {
                        if (errno == EAGAIN || errno == EWOULDBLOCK) {
                            // No connection available, continue
                            continue;
                        }
                        if (running.load()) {
                            std::cout << "DEBUG: Process " << process_id << " failed to accept connection (errno: " << errno << ")" << std::endl;
                            std::cout << "DEBUG: Process " << process_id << " errno description: " << strerror(errno) << std::endl;
                        }
                        continue;
                    }
                    
                    connection_count++;
                    std::cout << "DEBUG: Process " << process_id << " accepted connection #" << connection_count << " from client socket " << client_socket << std::endl;
                    std::cout << "DEBUG: Process " << process_id << " client address: " << inet_ntoa(client_addr.sin_addr) << ":" << ntohs(client_addr.sin_port) << std::endl;
                    
                    // Process the connection immediately
                    std::cout << "DEBUG: Process " << process_id << " processing connection " << client_socket << std::endl;
                    process_connection(client_socket);
                    std::cout << "DEBUG: Process " << process_id << " finished processing connection " << client_socket << std::endl;
                } catch (const std::exception& e) {
                    std::cout << "DEBUG: Process " << process_id << " exception in message receiver: " << e.what() << std::endl;
                    if (running.load()) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    }
                } catch (...) {
                    std::cout << "DEBUG: Process " << process_id << " unknown exception in message receiver" << std::endl;
                    if (running.load()) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    }
                }
            }
            std::cout << "DEBUG: Process " << process_id << " message receiver thread exiting after " << iteration_count << " iterations" << std::endl;
        });
        
        std::cout << "DEBUG: Process " << process_id << " message receiver thread created successfully" << std::endl;
        
        // Give the thread a moment to start and validate it's running
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        if (receive_thread.joinable()) {
            std::cout << "DEBUG: Process " << process_id << " message receiver thread is joinable (running)" << std::endl;
            
            // Test if we can send a message to ourselves to verify the receiver is working
            // DISABLED: Self-test causes segfault due to race condition
            // std::cout << "DEBUG: Process " << process_id << " testing message receiver with self-test message" << std::endl;
            // Message test_msg(XFER, process_id, process_id, 0, 1);
            // if (send_message(process_id, test_msg)) {
            //     std::cout << "DEBUG: Process " << process_id << " self-test message sent successfully" << std::endl;
            // } else {
            //     std::cout << "DEBUG: Process " << process_id << " self-test message failed" << std::endl;
            // }
        } else {
            std::cerr << "ERROR: Process " << process_id << " message receiver thread is not joinable (failed to start)" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Process " << process_id << " failed to create message receiver thread: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "ERROR: Process " << process_id << " unknown exception while creating message receiver thread" << std::endl;
    }
}

void NetworkManager::process_connection(int client_socket) {
    try {
        // Set client socket to non-blocking mode for timeout control
        int flags = fcntl(client_socket, F_GETFL, 0);
        if (flags >= 0) {
            fcntl(client_socket, F_SETFL, flags | O_NONBLOCK);
        }
        
        // Receive message length with timeout
        size_t len = 0;
        std::cout << "DEBUG: Process " << process_id << " attempting to receive message length from socket " << client_socket << std::endl;
        
        // Use select() to wait for data with timeout
        fd_set read_fds;
        FD_ZERO(&read_fds);
        FD_SET(client_socket, &read_fds);
        
        struct timeval timeout;
        timeout.tv_sec = 1;  // 1 second timeout
        timeout.tv_usec = 0;
        
        int select_result = select(client_socket + 1, &read_fds, NULL, NULL, &timeout);
        if (select_result <= 0) {
            std::cout << "DEBUG: Process " << process_id << " timeout waiting for message length from socket " << client_socket << std::endl;
            close(client_socket);
            return;
        }
        
        // Set socket back to blocking for receive
        flags = fcntl(client_socket, F_GETFL, 0);
        if (flags >= 0) {
            fcntl(client_socket, F_SETFL, flags & ~O_NONBLOCK);
        }
        
        ssize_t bytes_received = recv(client_socket, &len, sizeof(len), MSG_NOSIGNAL);
        if (bytes_received <= 0) {
            if (bytes_received == 0) {
                std::cout << "DEBUG: Process " << process_id << " connection closed by peer on socket " << client_socket << std::endl;
            } else {
                std::cout << "DEBUG: Process " << process_id << " failed to receive message length from socket " << client_socket << " (errno: " << errno << ")" << std::endl;
            }
            close(client_socket);
            return;
        }
        
        std::cout << "DEBUG: Process " << process_id << " received message length: " << len << " from socket " << client_socket << std::endl;
        
        // Validate message length
        if (len > 1024 * 1024) { // 1MB limit
            std::cout << "DEBUG: Process " << process_id << " message too large: " << len << " bytes, closing connection" << std::endl;
            close(client_socket);
            return;
        }
        
        // Receive message data
        std::vector<char> buffer(len + 1);
        bytes_received = recv(client_socket, buffer.data(), len, MSG_NOSIGNAL);
        if (bytes_received <= 0) {
            if (bytes_received == 0) {
                std::cout << "DEBUG: Process " << process_id << " connection closed by peer during data receive" << std::endl;
            } else {
                std::cout << "DEBUG: Process " << process_id << " failed to receive message data (errno: " << errno << ")" << std::endl;
            }
            close(client_socket);
            return;
        }
        
        // Validate we received the expected amount of data
        if (static_cast<size_t>(bytes_received) != len) {
            std::cout << "DEBUG: Process " << process_id << " incomplete message received: expected " << len << " bytes, got " << bytes_received << std::endl;
            close(client_socket);
            return;
        }
        
        buffer[len] = '\0';
        std::cout << "DEBUG: Process " << process_id << " received message data from socket " << client_socket << ": " << std::string(buffer.data()) << std::endl;
        
        Message msg = Message::deserialize(std::string(buffer.data()));
        std::cout << "DEBUG: Process " << process_id << " deserialized message from socket " << client_socket << ": type=" << static_cast<int>(msg.type) << " from=" << msg.from << " to=" << msg.to << " value=" << msg.value << std::endl;
        
        // Special debugging for TERM messages
        if (msg.type == TERM) {
            std::cout << "DEBUG: *** TERM MESSAGE RECEIVED *** Process " << process_id << " received TERM message from Process " << msg.from << std::endl;
            std::cout << "DEBUG: *** TERM MESSAGE RECEIVED *** Message details - type=" << static_cast<int>(msg.type) << ", from=" << msg.from << ", to=" << msg.to << ", value=" << msg.value << ", snapshot_id=" << msg.snapshot_id << std::endl;
        }
        
        // Validate message
        if (msg.type == UNKNOWN) {
            std::cout << "DEBUG: Process " << process_id << " received invalid message, ignoring" << std::endl;
            close(client_socket);
            return;
        }
        
        // Queue the message for processing by receive_function
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            message_queue.push(msg);
            std::cout << "DEBUG: Process " << process_id << " queued message: type=" << static_cast<int>(msg.type) << " from=" << msg.from << " to=" << msg.to << " value=" << msg.value << " (queue size: " << message_queue.size() << ")" << std::endl;
            
            // Special debugging for TERM messages being queued
            if (msg.type == TERM) {
                std::cout << "DEBUG: *** TERM MESSAGE QUEUED *** Process " << process_id << " queued TERM message from Process " << msg.from << std::endl;
                std::cout << "DEBUG: *** TERM MESSAGE QUEUED *** Queue size: " << message_queue.size() << std::endl;
            }
        }
        
        // Close the connection after processing one message
        close(client_socket);
        std::cout << "DEBUG: Process " << process_id << " closed connection " << client_socket << " after processing message" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "DEBUG: Process " << process_id << " exception in process_connection: " << e.what() << std::endl;
        close(client_socket);
    } catch (...) {
        std::cout << "DEBUG: Process " << process_id << " unknown exception in process_connection" << std::endl;
        close(client_socket);
    }
}

void NetworkManager::stop_message_receiver() {
    std::cout << "DEBUG: Process " << process_id << " stopping message receiver thread" << std::endl;
    running = false;
    if (receive_thread.joinable()) {
        receive_thread.join();
    }
}

bool NetworkManager::has_message() {
    std::lock_guard<std::mutex> lock(queue_mutex);
    return !message_queue.empty();
}

Message NetworkManager::get_next_message() {
    return receive_message();
}

int NetworkManager::get_queue_size() {
    std::lock_guard<std::mutex> lock(queue_mutex);
    return message_queue.size();
}

void NetworkManager::set_message_processor_callback(std::function<void(const Message&)> callback) {
    message_processor_callback = callback;
}

void Logger::log_send(const Message& msg, const std::string& extra) {
    std::lock_guard<std::mutex> lock(log_mutex);
    if (log_file.is_open()) {
        std::string timestamp = get_timestamp();
        log_file << "Process " << process_id << " sends " << msg.value 
                 << " units to Process " << msg.to << " at " << timestamp;
        if (!extra.empty()) {
            log_file << " " << extra;
        }
        log_file << std::endl;
        log_file.flush();
    }
}

void Logger::log_receive(const Message& msg, const std::string& extra) {
    std::lock_guard<std::mutex> lock(log_mutex);
    if (log_file.is_open()) {
        std::string timestamp = get_timestamp();
        log_file << "Process " << process_id << " receives " << msg.value 
                 << " units from Process " << msg.from << " at " << timestamp;
        if (!extra.empty()) {
            log_file << " " << extra;
        }
        log_file.flush();
    }
}

void Logger::log_snapshot(int snapshot_id, int balance, const std::string& action) {
    std::lock_guard<std::mutex> lock(log_mutex);
    if (log_file.is_open()) {
        std::string timestamp = get_timestamp();
        log_file << "Process " << process_id << " " << action 
                 << " snapshot " << snapshot_id << " with balance " << balance 
                 << " at " << timestamp << std::endl;
        log_file.flush();
    }
}

void Logger::log_general(const std::string& message) {
    std::lock_guard<std::mutex> lock(log_mutex);
    if (log_file.is_open()) {
        std::string timestamp = get_timestamp();
        log_file << "[" << timestamp << "] " << message << std::endl;
        log_file.flush();
    }
}

void Logger::log_snapshot_initiation(int snapshot_id, const std::string& target_processes) {
    std::lock_guard<std::mutex> lock(log_mutex);
    if (log_file.is_open()) {
        std::string timestamp = get_timestamp();
        log_file << "[" << timestamp << "] Coordinator initiates snapshot " << snapshot_id 
                 << " to processes: " << target_processes << std::endl;
        log_file.flush();
    }
}

void Logger::log_snapshot_completion(int snapshot_id, int total_balance, int process_balances, int channel_balances) {
    std::lock_guard<std::mutex> lock(log_mutex);
    if (log_file.is_open()) {
        std::string timestamp = get_timestamp();
        log_file << "[" << timestamp << "] Coordinator completes snapshot " << snapshot_id 
                 << " with total balance " << total_balance 
                 << " (processes: " << process_balances << ", channels: " << channel_balances << ")" << std::endl;
        log_file.flush();
    }
}

void Logger::log_snapshot_analysis(int snapshot_id, bool is_consistent, int expected_total, int actual_total) {
    std::lock_guard<std::mutex> lock(log_mutex);
    if (log_file.is_open()) {
        std::string timestamp = get_timestamp();
        log_file << "[" << timestamp << "] Snapshot " << snapshot_id << " analysis: " 
                 << (is_consistent ? "CONSISTENT" : "INCONSISTENT")
                 << " (expected: " << expected_total << ", actual: " << actual_total << ")" << std::endl;
        log_file.flush();
    }
}

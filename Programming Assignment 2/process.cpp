#include "process.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <random>
#include <set>

Process::Process(int pid, int initial_bal, const std::vector<int>& neigh, int n_proc, bool is_coord, int k, double lambda_c_val, int exp_id, const std::string& topology, const std::vector<int>& all_initial_balances)
    : process_id(pid), initial_balance(initial_bal), current_balance(initial_bal),
      neighbors(neigh), original_topology_neighbors(neigh), n_processes(n_proc), should_terminate(false), coordinator_termination_sent(false), is_coordinator(is_coord),
      k_snapshots(k), completed_snapshots(0), lambda_c(lambda_c_val), gen(rd()), exp_dist(1.0), initial_balances(all_initial_balances) {
    
    std::cout << "DEBUG: Process " << pid << " constructor started" << std::endl;
    network = new NetworkManager(process_id, 8000, exp_id);
    std::cout << "DEBUG: Process " << pid << " NetworkManager created" << std::endl;
    logger = new Logger(process_id, is_coordinator, topology, n_proc);
    
    // Calculate initial total balance
    if (is_coordinator) {
        // Coordinator (Process 0) has no balance, but we need to calculate the total system balance
        // This will be calculated dynamically when we receive the first snapshot replies
        // For now, set to 0 and it will be updated during snapshot analysis
        initial_total_balance = 0;
    } else {
        // For non-coordinator processes, this is just their individual balance
        initial_total_balance = initial_bal;
    }
    
    logger->log_general("Process " + std::to_string(process_id) + 
                       " started with initial balance " + std::to_string(initial_balance));
    
    
    if (is_coordinator) {
        // Calculate expected total balance from all non-coordinator processes
        // This will be updated when we receive snapshot replies
        logger->log_general("Coordinator (Process 0) - no balance, will track system total from snapshots");
    }
}

Process::~Process() {
    stop();
    delete network;
    delete logger;
}

void Process::start_server() {
    std::cout << "DEBUG: Process " << process_id << " start_server() called" << std::endl;
    logger->log_general("DEBUG: Process " + std::to_string(process_id) + " start_server() called");
    
    if (!network->initialize_server()) {
        std::cerr << "Process " << process_id << ": Failed to initialize server" << std::endl;
        logger->log_general("ERROR: Process " + std::to_string(process_id) + " failed to initialize server");
        return;
    }
    
    std::cout << "Process " << process_id << " server started" << std::endl;
    logger->log_general("Process " + std::to_string(process_id) + " server started");
}

void Process::start_connections() {
    logger->log_general("DEBUG: Process " + std::to_string(process_id) + " starting connections phase");
    
    // Start message receiver BEFORE establishing connections
    // This allows us to accept incoming connections while we're establishing outgoing ones
    logger->log_general("DEBUG: Process " + std::to_string(process_id) + " starting message receiver");
    std::cout << "DEBUG: Process " << process_id << " about to start message receiver" << std::endl;
    network->start_message_receiver();
    std::cout << "DEBUG: Process " << process_id << " message receiver started successfully" << std::endl;
    
    // Now establish outgoing connections
    if (!network->initialize_connections(neighbors)) {
        std::cerr << "Process " << process_id << ": Failed to initialize connections" << std::endl;
        logger->log_general("ERROR: Process " + std::to_string(process_id) + " failed to initialize connections");
        return;
    }
    
    logger->log_general("DEBUG: Process " + std::to_string(process_id) + " connections initialized successfully");
    std::cout << "DEBUG: Process " << process_id << " connections initialized successfully" << std::endl;
    
    logger->log_general("DEBUG: Process " + std::to_string(process_id) + " constructor completed successfully");
    std::cout << "DEBUG: Process " << process_id << " constructor completed successfully" << std::endl;
    
    should_terminate = false;
    logger->log_general("DEBUG: Process " + std::to_string(process_id) + " should_terminate set to false");
    
    if (is_coordinator) {
        logger->log_general("DEBUG: Process " + std::to_string(process_id) + " starting coordinator send thread");
        send_thread = std::thread(&Process::coordinator_send_function, this);
    } else {
        logger->log_general("DEBUG: Process " + std::to_string(process_id) + " starting regular send thread");
        send_thread = std::thread(&Process::send_function, this);
    }
    
    // Start receive thread for queue-based message processing
    receive_thread = std::thread(&Process::receive_function, this);
    logger->log_general("DEBUG: Process " + std::to_string(process_id) + " started receive thread");
    
    // Give threads a moment to start
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Verify threads are running
    if (send_thread.joinable()) {
        logger->log_general("DEBUG: Process " + std::to_string(process_id) + " send thread is running");
    } else {
        logger->log_general("ERROR: Process " + std::to_string(process_id) + " send thread failed to start");
    }
    
    if (receive_thread.joinable()) {
        logger->log_general("DEBUG: Process " + std::to_string(process_id) + " receive thread is running");
    } else {
        logger->log_general("ERROR: Process " + std::to_string(process_id) + " receive thread failed to start");
    }
    
    std::cout << "Process " << process_id << " started" << std::endl;
    logger->log_general("Process " + std::to_string(process_id) + " started successfully");
    
    // Log connection status
    logger->log_general("Process " + std::to_string(process_id) + " network initialization completed");
    
    // Log successful startup
    logger->log_general("Process " + std::to_string(process_id) + " fully started and ready");
    logger->log_general("DEBUG: Process " + std::to_string(process_id) + " all threads started, entering main loop");
}

void Process::start() {
    // This method is kept for backward compatibility
    start_server();
    start_connections();
}

void Process::stop() {
    logger->log_general("DEBUG: Process " + std::to_string(process_id) + " stop() called - setting should_terminate = true");
    should_terminate = true;
    
    if (send_thread.joinable()) {
        send_thread.join();
    }
    
    if (receive_thread.joinable()) {
        receive_thread.join();
    }
    
    // Stop the message receiver
    network->stop_message_receiver();
    network->cleanup();
    std::cout << "Process " << process_id << " stopped" << std::endl;
}


void Process::send_function() {
    logger->log_general("DEBUG: Process " + std::to_string(process_id) + " send function started");
    
    // Coordinator doesn't participate in money transfers - it only handles snapshots
    if (is_coordinator) {
        logger->log_general("DEBUG: Process " + std::to_string(process_id) + " is coordinator, send function exiting immediately");
        logger->log_general("Coordinator send function started");
        logger->log_general("Coordinator send function started");
        return;
    }
    
    logger->log_general("DEBUG: Process " + std::to_string(process_id) + " entering send loop");
    
    while (!should_terminate.load()) {
        logger->log_general("DEBUG: Process " + std::to_string(process_id) + " send loop iteration, should_terminate=" + std::to_string(should_terminate.load()));
        
        // Exponential sleep with mean lambda
        int sleep_time = get_exponential_random(5.0); // lambda = 5
        logger->log_general("DEBUG: Process " + std::to_string(process_id) + " sleeping for " + std::to_string(sleep_time) + "ms");
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
        
        if (should_terminate.load()) {
            logger->log_general("DEBUG: Process " + std::to_string(process_id) + " send_function: should_terminate is true, breaking send loop");
            break;
        }
        
        // Check if we should stop sending (e.g., if coordinator has finished all snapshots)
        if (is_coordinator) {
            // Coordinator doesn't send money transfers
            continue;
        }
        
        // Get current balance
        int curr_bal = current_balance.load();
        logger->log_general("DEBUG: Process " + std::to_string(process_id) + " current balance: " + std::to_string(curr_bal));
        
        // If balance is zero, continue (wait for funds)
        if (curr_bal == 0) {
            logger->log_general("DEBUG: Process " + std::to_string(process_id) + " balance is 0, continuing");
            continue;
        }
        
        // Choose random neighbor from original topology only
        if (original_topology_neighbors.empty()) {
            logger->log_general("DEBUG: Process " + std::to_string(process_id) + " no original topology neighbors, continuing");
            continue;
        }
        
        std::uniform_int_distribution<> dis(0, original_topology_neighbors.size() - 1);
        int pj = original_topology_neighbors[dis(gen)];
        if (pj == process_id) {
            logger->log_general("DEBUG: Process " + std::to_string(process_id) + " selected self as neighbor, continuing");
            continue; // Can't send to self
        }
        
        // Choose random amount between 1 and current balance
        int rand_val = get_random_amount();
        if (rand_val <= 0) {
            logger->log_general("DEBUG: Process " + std::to_string(process_id) + " random amount is " + std::to_string(rand_val) + ", continuing");
            continue;
        }
        
        logger->log_general("DEBUG: Process " + std::to_string(process_id) + " attempting to send " + std::to_string(rand_val) + " to Process " + std::to_string(pj));
        
        // Update balance first
        current_balance -= rand_val;
        
        // Send transfer message
        Message msg(XFER, process_id, pj, 0, rand_val);
        if (network->send_message(pj, msg)) {
            logger->log_send(msg);
            logger->log_general("DEBUG: Process " + std::to_string(process_id) + " successfully sent message to Process " + std::to_string(pj));
        } else {
            // Restore balance if send failed
            current_balance += rand_val;
            logger->log_general("DEBUG: Process " + std::to_string(process_id) + " failed to send message to Process " + std::to_string(pj) + ", restored balance");
        }
    }
    
    logger->log_general("DEBUG: Process " + std::to_string(process_id) + " send function exiting");
}

void Process::receive_function() {
    logger->log_general("DEBUG: Process " + std::to_string(process_id) + " receive function started");
    logger->log_general("DEBUG: Process " + std::to_string(process_id) + " receive_function initial state: should_terminate=" + std::to_string(should_terminate.load()) + 
                       ", is_coordinator=" + std::to_string(is_coordinator) + 
                       ", coordinator_termination_sent=" + std::to_string(coordinator_termination_sent.load()));
    logger->log_general("Process " + std::to_string(process_id) + " will stay alive until TERM message is received");
    
    // Special logging for coordinator
    if (is_coordinator) {
        logger->log_general("DEBUG: Coordinator receive function is running and ready to process messages");
    } else {
        logger->log_general("DEBUG: Non-coordinator Process " + std::to_string(process_id) + " receive function will stay alive until TERM message is received");
    }
    
    // Add timeout for non-coordinator processes
    auto start_time = std::chrono::high_resolution_clock::now();
    const int timeout_seconds = 60; // 1 minute timeout (reduced for faster testing)
    
    // Keep receive function running for non-coordinator processes until they receive TERM message
    // For coordinator, keep running until termination is sent
    // Non-coordinator processes should keep running until they explicitly receive and process a TERM message
    int receive_iteration = 0;
    bool term_message_processed = false;
    while ((!should_terminate.load() || !term_message_processed) && (is_coordinator ? !coordinator_termination_sent.load() : true)) {
        receive_iteration++;
        
        // Reduce logging frequency to improve performance
        if (receive_iteration % 100 == 0) {
            logger->log_general("DEBUG: Process " + std::to_string(process_id) + " receive_function iteration " + std::to_string(receive_iteration) + 
                               ", should_terminate=" + std::to_string(should_terminate.load()) + 
                               ", coordinator_termination_sent=" + std::to_string(coordinator_termination_sent.load()));
        }
        
        try {
            // Check timeout for non-coordinator processes
            if (!is_coordinator) {
                auto current_time = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time);
                if (elapsed.count() >= timeout_seconds) {
                    logger->log_general("DEBUG: Process " + std::to_string(process_id) + " timeout reached (" + std::to_string(timeout_seconds) + "s), forcing termination");
                    logger->log_general("Process " + std::to_string(process_id) + " terminating due to timeout (no TERM message received)");
                    should_terminate = true;
                    term_message_processed = true;
                    return;
                }
                
                // Additional check: if coordinator has been terminated for a while, force termination
                // This is a more aggressive approach to handle the case where TERM messages are not delivered
                if (elapsed.count() >= 30) { // After 30 seconds
                    logger->log_general("DEBUG: Process " + std::to_string(process_id) + " forcing termination after 30 seconds (coordinator likely terminated)");
                    logger->log_general("Process " + std::to_string(process_id) + " terminating due to coordinator termination timeout");
                    should_terminate = true;
                    term_message_processed = true;
                    return;
                }
            }
            
            // Process messages in batches - always try to process messages
            int messages_processed = 0;
            int max_batch_size = 10; // Process up to 10 messages at once
            
            // Try to process messages until we've processed max_batch_size or no more messages
            while (messages_processed < max_batch_size) {
                // Always try to get a message - don't rely on queue size check due to race conditions
                Message msg = network->receive_message();
                
                // Check if we got an empty message (queue was empty)
                if (msg.type == UNKNOWN) {
                    break;
                }
                
                // Log queue size when we start processing
                if (messages_processed == 0) {
                    int queue_size = network->get_queue_size();
                    logger->log_general("DEBUG: Process " + std::to_string(process_id) + " processing message from queue (size: " + std::to_string(queue_size) + ")...");
                }
                
                messages_processed++;
                
                logger->log_general("DEBUG: Process " + std::to_string(process_id) + " processing message #" + std::to_string(messages_processed) + " in batch: type=" + std::to_string(static_cast<int>(msg.type)) + ", from=" + std::to_string(msg.from) + ", to=" + std::to_string(msg.to) + ", value=" + std::to_string(msg.value));
                
                // Always process TERM messages, even if should_terminate is already true
                if (msg.type == TERM && !is_coordinator) {
                    logger->log_general("DEBUG: Process " + std::to_string(process_id) + " received TERM message, terminating");
                    logger->log_general("Received termination message from Process " + std::to_string(msg.from));
                    logger->log_general("Process " + std::to_string(process_id) + " terminating gracefully");
                    logger->log_general("DEBUG: Process " + std::to_string(process_id) + " setting should_terminate = true due to TERM message");
                    should_terminate = true;
                    term_message_processed = true;
                    return;
                }
                
                if (should_terminate.load()) {
                    logger->log_general("DEBUG: Process " + std::to_string(process_id) + " receive_function: should_terminate is true, breaking message processing loop");
                    break;
                }
                
                // Check if message is valid (not empty)
                if (msg.type == UNKNOWN) {
                    logger->log_general("DEBUG: Process " + std::to_string(process_id) + " received UNKNOWN message, continuing");
                    continue;
                }
                
                logger->log_general("DEBUG: Process " + std::to_string(process_id) + " received message type " + std::to_string(static_cast<int>(msg.type)) + " from " + std::to_string(msg.from) + " (batch " + std::to_string(messages_processed) + "/" + std::to_string(max_batch_size) + ")");
                // Log the received message
                logger->log_receive(msg);
                
                        switch (msg.type) { 
                        case XFER: {
                            logger->log_general("DEBUG: Process " + std::to_string(process_id) + " handling XFER message");
                            // All processes handle money transfers (both sending and receiving)
                            int old_balance = current_balance.load();
                            current_balance += msg.value;
                            int new_balance = current_balance.load();
                            logger->log_general("DEBUG: Process " + std::to_string(process_id) + " balance update: " + std::to_string(old_balance) + " + " + std::to_string(msg.value) + " = " + std::to_string(new_balance) + " (from Process " + std::to_string(msg.from) + ")");
                            
                            // Record XFER message in channel states if we're recording for any snapshot
                            std::lock_guard<std::mutex> lock(snapshot_mutex);
                            for (auto& snapshot_pair : channel_recording) {
                                int snapshot_id = snapshot_pair.first;
                                for (auto& channel_pair : snapshot_pair.second) {
                                    int neighbor = channel_pair.first;
                                    bool is_recording = channel_pair.second;
                                    if (is_recording && neighbor == msg.from) {
                                        channel_states[snapshot_id][neighbor].push_back(msg.value);
                                        logger->log_general("DEBUG: Process " + std::to_string(process_id) + " recorded XFER message " + std::to_string(msg.value) + " from Process " + std::to_string(msg.from) + " in channel state for snapshot " + std::to_string(snapshot_id));
                                    }
                                }
                            }
                            
                            break;
                        }
                    
                case MARKER:
                    logger->log_general("DEBUG: Process " + std::to_string(process_id) + " handling MARKER message");
                    // Handle snapshot marker
                    handle_marker(msg);
                    break;
                    
                case SNAP_REPLY:
                    logger->log_general("DEBUG: Process " + std::to_string(process_id) + " handling SNAP_REPLY message");
                    if (is_coordinator) {
                        logger->log_general("DEBUG: Coordinator processing SNAP_REPLY from Process " + std::to_string(msg.from) + " for snapshot " + std::to_string(msg.snapshot_id));
                        handle_snapshot_reply(msg);
                    }
                    break;
                    
                case TERM:
                    if (is_coordinator) {
                        logger->log_general("DEBUG: Coordinator received TERM message from Process " + std::to_string(msg.from) + " - ignoring until all snapshots complete");
                        logger->log_general("Coordinator will continue with snapshots and terminate after completion");
                    } else {
                        // TERM messages for non-coordinator processes are handled above in the early check
                        logger->log_general("DEBUG: Process " + std::to_string(process_id) + " TERM message already handled above");
                    }
                    break;
                    
                default:
                    logger->log_general("DEBUG: Process " + std::to_string(process_id) + " received unknown message type: " + std::to_string(static_cast<int>(msg.type)));
                    break;
                }
            } // End of batch processing loop
            
            // If no messages were processed, sleep briefly to prevent busy waiting
            if (messages_processed == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            } else {
                // If messages were processed, don't sleep - immediately try to process more
                // This ensures TERM messages are processed as quickly as possible
            }
        } catch (const std::exception& e) {
            logger->log_general("DEBUG: Process " + std::to_string(process_id) + " exception in receive: " + std::string(e.what()));
            logger->log_general("DEBUG: Process " + std::to_string(process_id) + " receive_function exception handler - should_terminate=" + std::to_string(should_terminate.load()));
            // Handle any network errors gracefully
            if (!should_terminate.load()) {
                logger->log_general("DEBUG: Process " + std::to_string(process_id) + " receive_function exception handler - continuing after exception");
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            } else {
                logger->log_general("DEBUG: Process " + std::to_string(process_id) + " receive_function exception handler - should_terminate=true, breaking out of loop");
                break;
            }
        } catch (...) {
            logger->log_general("DEBUG: Process " + std::to_string(process_id) + " unknown exception in receive");
            logger->log_general("DEBUG: Process " + std::to_string(process_id) + " receive_function unknown exception handler - should_terminate=" + std::to_string(should_terminate.load()));
            // Handle any network errors gracefully
            if (!should_terminate.load()) {
                logger->log_general("DEBUG: Process " + std::to_string(process_id) + " receive_function unknown exception handler - continuing after exception");
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            } else {
                logger->log_general("DEBUG: Process " + std::to_string(process_id) + " receive_function unknown exception handler - should_terminate=true, breaking out of loop");
                break;
            }
        }
    }
    
    logger->log_general("DEBUG: Process " + std::to_string(process_id) + " receive_function while loop exited. should_terminate=" + std::to_string(should_terminate.load()) + 
                       ", coordinator_termination_sent=" + std::to_string(coordinator_termination_sent.load()));
    logger->log_general("Process " + std::to_string(process_id) + " receive function terminating");
}

void Process::handle_marker(const Message& msg) {
    int snapshot_id = msg.snapshot_id;
    int from_neighbor = msg.from;
    
    logger->log_general("Received marker for snapshot " + std::to_string(snapshot_id) + 
                       " from Process " + std::to_string(from_neighbor));
    logger->log_general("Processing marker for snapshot " + std::to_string(snapshot_id));
    
    bool snapshot_complete = false;
    
    {
        std::lock_guard<std::mutex> lock(snapshot_mutex);
        
        // If this is the first marker for this snapshot, record local state
        if (!recorded_local_state[snapshot_id]) {
            logger->log_general("Recording local state for snapshot " + std::to_string(snapshot_id));
            record_local_state(snapshot_id);
            logger->log_general("Starting channel recording for snapshot " + std::to_string(snapshot_id));
            start_channel_recording(snapshot_id);
            
            // Log snapshot recording
            logger->log_snapshot(snapshot_id, current_balance.load(), "recording snapshot with current balance");
            logger->log_general("Completed local state recording for snapshot " + std::to_string(snapshot_id));
        }
        
        // Stop recording on the channel where marker arrived
        logger->log_general("Stopping channel recording for snapshot " + std::to_string(snapshot_id) + " from Process " + std::to_string(from_neighbor));
        stop_channel_recording(snapshot_id, from_neighbor);
        logger->log_general("Stopped channel recording for snapshot " + std::to_string(snapshot_id));
        
        // Check if snapshot is complete
        snapshot_complete = is_snapshot_complete(snapshot_id);
    }
    
    // Send marker to all other neighbors (outside mutex lock)
    logger->log_general("Sending markers to other neighbors for snapshot " + std::to_string(snapshot_id));
    for (int neighbor : neighbors) {
        if (neighbor != from_neighbor && neighbor != process_id) {
            Message marker_msg(MARKER, process_id, neighbor, snapshot_id, 0);
            bool marker_sent = network->send_message(neighbor, marker_msg);
            if (marker_sent) {
                logger->log_general("Sent marker for snapshot " + std::to_string(snapshot_id) + " to Process " + std::to_string(neighbor));
            } else {
                logger->log_general("Failed to send marker for snapshot " + std::to_string(snapshot_id) + " to Process " + std::to_string(neighbor));
            }
        }
    }
    
    // Send snapshot reply if complete (outside mutex lock to avoid deadlock)
    if (snapshot_complete) {
        logger->log_general("Snapshot " + std::to_string(snapshot_id) + " complete, sending reply to coordinator");
        send_snapshot_reply(snapshot_id);
    } else {
        logger->log_general("Snapshot " + std::to_string(snapshot_id) + " not complete yet, waiting for more markers");
    }
    
    logger->log_general("Finished processing marker for snapshot " + std::to_string(snapshot_id));
}

void Process::handle_snapshot_reply(const Message& msg) {
    std::lock_guard<std::mutex> lock(snapshot_mutex);
    
    int snapshot_id = msg.snapshot_id;
    int process_id_reply = msg.from;
    int value = msg.value;
    
    if (value >= 0) {
        // This is a process balance (positive value)
        logger->log_general("DEBUG: Coordinator received snapshot reply from Process " + std::to_string(process_id_reply) + 
                           " for snapshot " + std::to_string(snapshot_id) + " with balance " + std::to_string(value));
        
        snapshot_collection[snapshot_id][process_id_reply] = value;
        snapshot_pending[snapshot_id].erase(process_id_reply);
        
        logger->log_general("DEBUG: Coordinator updated snapshot collection for snapshot " + std::to_string(snapshot_id) + 
                           ". Remaining pending processes: " + std::to_string(snapshot_pending[snapshot_id].size()));
        
        logger->log_general("Received snapshot reply from Process " + std::to_string(process_id_reply) + 
                           " for snapshot " + std::to_string(snapshot_id) + " with balance " + std::to_string(value));
        
        // Check if all replies received
        if (snapshot_pending[snapshot_id].empty() && !snapshot_analyzed[snapshot_id]) {
            logger->log_general("DEBUG: Coordinator received all snapshot replies for snapshot " + std::to_string(snapshot_id) + ", analyzing...");
            snapshot_analyzed[snapshot_id] = true;
            analyze_snapshot(snapshot_id);
        } else if (!snapshot_pending[snapshot_id].empty()) {
            logger->log_general("DEBUG: Coordinator still waiting for " + std::to_string(snapshot_pending[snapshot_id].size()) + " more snapshot replies for snapshot " + std::to_string(snapshot_id));
        } else {
            logger->log_general("DEBUG: Snapshot " + std::to_string(snapshot_id) + " already analyzed, ignoring duplicate reply");
        }
    } else {
        // This is a channel message (negative value)
        int channel_value = -value; // Convert back to positive
        snapshot_channels[snapshot_id][process_id_reply].push_back(channel_value);
        logger->log_general("DEBUG: Coordinator received channel message " + std::to_string(channel_value) + " from Process " + std::to_string(process_id_reply) + " for snapshot " + std::to_string(snapshot_id));
    }
}

void Process::record_local_state(int snapshot_id) {
    int balance = current_balance.load();
    recorded_local_state[snapshot_id] = true;
    recorded_balance[snapshot_id] = balance;  // Store the balance at the time of recording
    logger->log_snapshot(snapshot_id, balance, "recorded local state for");
}

void Process::start_channel_recording(int snapshot_id) {
    for (int neighbor : neighbors) {
        if (neighbor != process_id) {
            channel_recording[snapshot_id][neighbor] = true;
            channel_states[snapshot_id][neighbor] = std::vector<int>();
        }
    }
}

void Process::stop_channel_recording(int snapshot_id, int from_neighbor) {
    channel_recording[snapshot_id][from_neighbor] = false;
    marker_received_on_channel[snapshot_id][from_neighbor] = true;
}

void Process::send_snapshot_reply(int snapshot_id) {
    if (is_coordinator) return;
    
    // Check if reply has already been sent to prevent duplicate replies
    {
        std::lock_guard<std::mutex> lock(snapshot_mutex);
        if (snapshot_reply_sent[snapshot_id]) {
            logger->log_general("DEBUG: Process " + std::to_string(process_id) + " snapshot reply for snapshot " + std::to_string(snapshot_id) + " already sent, skipping");
            return;
        }
        snapshot_reply_sent[snapshot_id] = true;
    }
    
    // Use the recorded balance at the time of snapshot, not the current balance
    int balance = recorded_balance[snapshot_id];
    logger->log_general("DEBUG: Process " + std::to_string(process_id) + " preparing to send snapshot reply for snapshot " + std::to_string(snapshot_id) + " with recorded balance " + std::to_string(balance));
    
    // Send process balance
    Message reply(SNAP_REPLY, process_id, 0, snapshot_id, balance); // Coordinator is process 0
    logger->log_general("DEBUG: Process " + std::to_string(process_id) + " sending snapshot reply to Process 0");
    
    bool reply_sent = network->send_message(0, reply);
    if (reply_sent) {
        logger->log_general("DEBUG: Process " + std::to_string(process_id) + " successfully sent snapshot reply to Process 0");
        logger->log_general("Sent snapshot reply for snapshot " + std::to_string(snapshot_id) + 
                           " with balance " + std::to_string(balance));
    } else {
        logger->log_general("DEBUG: Process " + std::to_string(process_id) + " FAILED to send snapshot reply to Process 0");
    }
    
    logger->log_general("DEBUG: Process " + std::to_string(process_id) + " balance sent successfully, now processing channel states");
    
    // Send channel states recorded during Chandy-Lamport snapshot
    // Make a copy of channel states to avoid holding the mutex for a long time
    logger->log_general("DEBUG: Process " + std::to_string(process_id) + " about to acquire snapshot_mutex for channel states");
    std::map<int, std::vector<int>> channel_states_copy;
    {
        std::lock_guard<std::mutex> lock(snapshot_mutex);
        logger->log_general("DEBUG: Process " + std::to_string(process_id) + " acquired snapshot_mutex, checking channel states for snapshot " + std::to_string(snapshot_id));
        channel_states_copy = channel_states[snapshot_id];
        logger->log_general("DEBUG: Process " + std::to_string(process_id) + " channel_states map size: " + std::to_string(channel_states_copy.size()));
    }
    
    // Now send the channel states without holding the mutex
    for (const auto& channel_pair : channel_states_copy) {
        int neighbor = channel_pair.first;
        const std::vector<int>& messages = channel_pair.second;
        
        logger->log_general("DEBUG: Process " + std::to_string(process_id) + " sending channel state for neighbor " + std::to_string(neighbor) + " with " + std::to_string(messages.size()) + " messages");
        
        // Send each message in the channel as a separate SNAP_REPLY with negative value to indicate it's a channel message
        for (int message_value : messages) {
            Message channel_reply(SNAP_REPLY, process_id, 0, snapshot_id, -message_value); // Negative value indicates channel message
            bool channel_sent = network->send_message(0, channel_reply);
            if (channel_sent) {
                logger->log_general("DEBUG: Process " + std::to_string(process_id) + " sent channel message " + std::to_string(message_value) + " for snapshot " + std::to_string(snapshot_id));
            } else {
                logger->log_general("DEBUG: Process " + std::to_string(process_id) + " FAILED to send channel message " + std::to_string(message_value) + " for snapshot " + std::to_string(snapshot_id));
            }
        }
    }
}

bool Process::is_snapshot_complete(int snapshot_id) {
    for (int neighbor : neighbors) {
        if (neighbor != process_id && !marker_received_on_channel[snapshot_id][neighbor]) {
            logger->log_general("Snapshot " + std::to_string(snapshot_id) + " not complete - waiting for marker from Process " + std::to_string(neighbor));
            return false;
        }
    }
    logger->log_general("Snapshot " + std::to_string(snapshot_id) + " is complete - all markers received");
    return true;
}

void Process::initiate_snapshot(int snapshot_id) {
    if (!is_coordinator) return;
    
    // Log snapshot initiation
    std::string target_processes;
    for (int neighbor : neighbors) {
        if (neighbor != process_id) {
            if (!target_processes.empty()) target_processes += ", ";
            target_processes += std::to_string(neighbor);
        }
    }
    logger->log_snapshot_initiation(snapshot_id, target_processes);
    
    // Record local state
    record_local_state(snapshot_id);
    start_channel_recording(snapshot_id);
    
    // Send markers to all neighbors
    for (int neighbor : neighbors) {
        if (neighbor != process_id) {
            Message marker_msg(MARKER, process_id, neighbor, snapshot_id, 0);
            network->send_message(neighbor, marker_msg);
        }
    }
    
    // Initialize pending processes for this snapshot
    for (int i = 1; i <= n_processes; i++) {
        if (i != process_id) {
            snapshot_pending[snapshot_id].insert(i);
        }
    }
}

void Process::analyze_snapshot(int snapshot_id) {
    if (!is_coordinator) return;
    
    int process_balances = 0;
    int channel_balances = 0;
    
    // Sum up all process balances from snapshot collection
    for (const auto& pair : snapshot_collection[snapshot_id]) {
        process_balances += pair.second;
    }
    
    // Coordinator (Process 0) has no balance, so don't add it
    // The snapshot collection contains all non-coordinator process balances
    
    // Calculate channel balances (messages in transit)
    for (const auto& channel_pair : snapshot_channels[snapshot_id]) {
        for (int amount : channel_pair.second) {
            channel_balances += amount;
        }
    }
    
    int total_balance = process_balances + channel_balances;
    
    // Log snapshot completion
    logger->log_snapshot_completion(snapshot_id, total_balance, process_balances, channel_balances);
    
    // Calculate expected total balance from the actual initial balances
    // Sum all non-coordinator process initial balances
    int expected_total = 0;
    for (int i = 1; i <= n_processes; i++) {
        // Get initial balance for process i from the input parameters
        // We need to calculate this from the initial_balances vector
        if (i <= static_cast<int>(initial_balances.size())) {
            expected_total += initial_balances[i-1];
        }
    }
    
    // Sanity check: total should equal initial total
    bool is_consistent = (total_balance == expected_total);
    
    logger->log_snapshot_analysis(snapshot_id, is_consistent, expected_total, total_balance);
    
    // Log detailed breakdown for debugging
    logger->log_general("Snapshot " + std::to_string(snapshot_id) + " breakdown:");
    logger->log_general("  - Process balances: " + std::to_string(process_balances));
    logger->log_general("  - Channel balances: " + std::to_string(channel_balances));
    logger->log_general("  - Total: " + std::to_string(total_balance));
    logger->log_general("  - Expected: " + std::to_string(expected_total));
    logger->log_general("  - Consistency: " + std::string(is_consistent ? "PASS" : "FAIL"));
    
    completed_snapshots++;
    
    // If all snapshots completed, log completion but don't terminate here
    // Termination will be handled in coordinator_send_function
    if (completed_snapshots >= k_snapshots) {
        logger->log_general("DEBUG: Coordinator finished all snapshots, termination will be handled by coordinator_send_function");
    }
}

void Process::coordinator_send_function() {
    logger->log_general("DEBUG: Process " + std::to_string(process_id) + " coordinator_send_function started");
    
    if (!is_coordinator) {
        logger->log_general("DEBUG: Process " + std::to_string(process_id) + " is not coordinator, exiting coordinator_send_function");
        return;
    }
    
    logger->log_general("DEBUG: Coordinator send function started");
    logger->log_general("Coordinator send function started");
    
    // Wait a bit more to ensure all connections are established
    logger->log_general("DEBUG: Coordinator waiting 500ms for connections to stabilize");
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    logger->log_general("DEBUG: Coordinator starting snapshot algorithm with " + std::to_string(k_snapshots) + " snapshots");
    logger->log_general("Coordinator starting snapshot algorithm with " + std::to_string(k_snapshots) + " snapshots");
    
    int total_snap_time = 0;
    
    for (int snapshot_id = 1; snapshot_id <= k_snapshots; snapshot_id++) {
        logger->log_general("DEBUG: Coordinator starting snapshot " + std::to_string(snapshot_id) + " of " + std::to_string(k_snapshots));
        
        // Coordinator must complete all snapshots regardless of should_terminate status
        // Only check should_terminate for non-coordinator processes
        
        // Sleep before initiating snapshot (exponential with mean lambda_c)
        int sleep_time = get_exponential_random(lambda_c);
        if (snapshot_id == 1) {
            sleep_time = std::min(sleep_time, 1000); // Limit first snapshot delay to 1 second
        }
        logger->log_general("DEBUG: Coordinator sleeping for " + std::to_string(sleep_time) + "ms before snapshot " + std::to_string(snapshot_id));
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
        
        // Record snapshot start time
        auto snap_start_time = std::chrono::high_resolution_clock::now();
        
        // Choose a random process to start the snapshot
        int target_process;
        if (process_id == 0) {
            // Coordinator (Process 0) chooses from all non-coordinator processes
            std::uniform_int_distribution<> dis(1, n_processes);
            target_process = dis(gen);
        } else {
            // Non-coordinator processes use neighbor selection
            target_process = get_random_neighbor();
            if (target_process == process_id) {
                for (int neighbor : neighbors) {
                    if (neighbor != process_id) {
                        target_process = neighbor;
                        break;
                    }
                }
            }
        }
        
        // Log snapshot initiation
        logger->log_snapshot_initiation(snapshot_id, std::to_string(target_process));
        
        // Send markers to all processes to ensure complete snapshot coverage
        logger->log_general("DEBUG: Coordinator sending markers to all processes for snapshot " + std::to_string(snapshot_id));
        for (int target_process = 1; target_process <= n_processes; target_process++) {
            // Only send one marker per process per snapshot
            if (markers_sent_for_snapshot[snapshot_id].find(target_process) == markers_sent_for_snapshot[snapshot_id].end()) {
                logger->log_general("DEBUG: Coordinator sending marker to Process " + std::to_string(target_process) + " for snapshot " + std::to_string(snapshot_id));
                Message marker_msg(MARKER, process_id, target_process, snapshot_id, 0);
                bool marker_sent = network->send_message(target_process, marker_msg);
                if (!marker_sent) {
                    logger->log_general("DEBUG: Coordinator FAILED to send marker to Process " + std::to_string(target_process) + " for snapshot " + std::to_string(snapshot_id));
                    logger->log_general("Failed to send marker to Process " + std::to_string(target_process) + " for snapshot " + std::to_string(snapshot_id));
                } else {
                    logger->log_general("DEBUG: Coordinator successfully sent marker to Process " + std::to_string(target_process) + " for snapshot " + std::to_string(snapshot_id));
                    markers_sent_for_snapshot[snapshot_id].insert(target_process);
                }
                
                // Add delay between marker sends to avoid overwhelming target processes
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            } else {
                logger->log_general("DEBUG: Coordinator already sent marker to Process " + std::to_string(target_process) + " for snapshot " + std::to_string(snapshot_id) + ", skipping");
            }
        }
        
        // Record local state for this snapshot
        record_local_state(snapshot_id);
        start_channel_recording(snapshot_id);
        
        // Initialize pending processes for this snapshot
        if (process_id == 0) {
            // Coordinator waits for all non-coordinator processes
            for (int i = 1; i <= n_processes; i++) {
                snapshot_pending[snapshot_id].insert(i);
            }
        } else {
            // Non-coordinator processes wait for their neighbors
            for (int neighbor : neighbors) {
                if (neighbor != process_id) {
                    snapshot_pending[snapshot_id].insert(neighbor);
                }
            }
        }
        
        // Wait for snapshot completion with timeout
        auto snapshot_start_wait = std::chrono::high_resolution_clock::now();
        while (!snapshot_pending[snapshot_id].empty() && !should_terminate.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            
            // Process any pending messages while waiting for snapshot completion
            if (network->has_message()) {
                Message msg = network->receive_message();
                if (msg.type == SNAP_REPLY) {
                    logger->log_general("DEBUG: Coordinator received SNAP_REPLY in snapshot waiting loop");
                    handle_snapshot_reply(msg);
                }
            }
            
            // Timeout after 5 seconds to prevent getting stuck (increased timeout)
            auto current_wait_time = std::chrono::high_resolution_clock::now();
            auto wait_duration = std::chrono::duration_cast<std::chrono::seconds>(current_wait_time - snapshot_start_wait);
            if (wait_duration.count() >= 5) {
                logger->log_general("Snapshot " + std::to_string(snapshot_id) + " timeout after 5 seconds - proceeding with available data");
                logger->log_general("DEBUG: Snapshot " + std::to_string(snapshot_id) + " pending processes: " + std::to_string(snapshot_pending[snapshot_id].size()));
                break;
            }
        }
        
        // Record snapshot end time
        auto snap_end_time = std::chrono::high_resolution_clock::now();
        auto snap_comp_time = std::chrono::duration_cast<std::chrono::milliseconds>(snap_end_time - snap_start_time);
        total_snap_time += snap_comp_time.count();
        
        // Log snapshot completion
        logger->log_general("Coordinator completes snapshot " + std::to_string(snapshot_id) + 
                           " in " + std::to_string(snap_comp_time.count()) + " ms");
        
        // Analyze snapshot for correctness (only if not already analyzed)
        // Note: analyze_snapshot is called in handle_snapshot_reply when all replies are received
        // We only call it here if we timed out and didn't receive all replies
        if (!snapshot_pending[snapshot_id].empty() && !snapshot_analyzed[snapshot_id]) {
            logger->log_general("DEBUG: Snapshot " + std::to_string(snapshot_id) + " timed out, analyzing with partial data");
            snapshot_analyzed[snapshot_id] = true;
            analyze_snapshot(snapshot_id);
            // Clear pending set to prevent duplicate analysis
            snapshot_pending[snapshot_id].clear();
        } else if (snapshot_analyzed[snapshot_id]) {
            logger->log_general("DEBUG: Snapshot " + std::to_string(snapshot_id) + " already analyzed in handle_snapshot_reply");
        } else {
            logger->log_general("DEBUG: Snapshot " + std::to_string(snapshot_id) + " completed successfully");
        }
        
        // Log progress
        logger->log_general("Completed " + std::to_string(snapshot_id) + " of " + std::to_string(k_snapshots) + " snapshots");
    }
    
    // Compute average snapshot time
    double avg_snap_time = (k_snapshots > 0) ? (double)total_snap_time / k_snapshots : 0;
    logger->log_general("Average snapshot time: " + std::to_string(avg_snap_time) + " ms");
    
    // Send termination messages to all processes
    logger->log_general("All snapshots completed. Sending termination messages.");
    logger->log_general("DEBUG: Coordinator finished all snapshots, sending TERM messages");
    
    if (process_id == 0) {
        // Coordinator sends termination to all non-coordinator processes
        logger->log_general("DEBUG: Coordinator sending TERM messages to processes 1-" + std::to_string(n_processes));
        for (int i = 1; i <= n_processes; i++) {
            logger->log_general("DEBUG: Coordinator sending TERM message to Process " + std::to_string(i));
            Message term_msg(TERM, process_id, i, 0, 0);
            
            // Log message details
            logger->log_general("DEBUG: TERM message details - type=" + std::to_string(static_cast<int>(term_msg.type)) + 
                               ", from=" + std::to_string(term_msg.from) + 
                               ", to=" + std::to_string(term_msg.to) + 
                               ", value=" + std::to_string(term_msg.value) + 
                               ", snapshot_id=" + std::to_string(term_msg.snapshot_id));
            
            bool term_sent = network->send_message(i, term_msg);
            if (term_sent) {
                logger->log_general("DEBUG: Coordinator successfully sent TERM to Process " + std::to_string(i));
                // Give a small delay to ensure message delivery
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            } else {
                logger->log_general("DEBUG: Coordinator FAILED to send TERM to Process " + std::to_string(i));
                // Try again with a delay
                logger->log_general("DEBUG: Coordinator retrying TERM message to Process " + std::to_string(i));
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                term_sent = network->send_message(i, term_msg);
                if (term_sent) {
                    logger->log_general("DEBUG: Coordinator successfully sent TERM to Process " + std::to_string(i) + " on retry");
                } else {
                    logger->log_general("DEBUG: Coordinator FAILED to send TERM to Process " + std::to_string(i) + " on retry");
                }
            }
        }
        
        // Coordinator should wait for other processes to terminate, not terminate immediately
        logger->log_general("DEBUG: Coordinator waiting for other processes to terminate...");
        
        // Wait a reasonable amount of time for other processes to receive and process TERM messages
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
        
        logger->log_general("DEBUG: Coordinator waiting period completed, continuing to receive messages");
        
        // Keep the coordinator alive to receive any final messages
        // Wait for a reasonable amount of time for other processes to terminate
        logger->log_general("DEBUG: Coordinator waiting for other processes to terminate...");
        std::this_thread::sleep_for(std::chrono::milliseconds(3000));
        
        logger->log_general("DEBUG: Coordinator has completed all snapshots and sent TERM messages to all processes");
        logger->log_general("DEBUG: Coordinator waiting longer for TERM message delivery before terminating...");
        
        // Wait much longer to ensure TERM messages are delivered and processed
        // The coordinator needs to stay alive long enough for TERM messages to be delivered
        logger->log_general("DEBUG: Coordinator waiting 30 seconds to ensure TERM messages are delivered...");
        std::this_thread::sleep_for(std::chrono::milliseconds(30000));
        
        logger->log_general("DEBUG: Coordinator has waited long enough, now terminating");
        coordinator_termination_sent = true;
        logger->log_general("DEBUG: Coordinator setting should_terminate = true after waiting for TERM message delivery");
        should_terminate = true;
        
    } else {
        // This should never happen - coordinator_send_function should only be called by coordinator
        logger->log_general("ERROR: Non-coordinator Process " + std::to_string(process_id) + " called coordinator_send_function - this is incorrect!");
        logger->log_general("DEBUG: Non-coordinator Process " + std::to_string(process_id) + " exiting coordinator_send_function without doing anything");
    }
    
    logger->log_general("DEBUG: Process " + std::to_string(process_id) + " coordinator_send_function exiting");
}

int Process::get_exponential_random(double lambda) {
    return static_cast<int>(-lambda * log(1.0 - (double)gen() / gen.max()));
}

int Process::get_random_neighbor() {
    if (neighbors.empty()) return process_id;
    std::uniform_int_distribution<> dis(0, neighbors.size() - 1);
    return neighbors[dis(gen)];
}

int Process::get_random_amount() {
    int balance = current_balance.load();
    if (balance <= 0) return 0;
    std::uniform_int_distribution<> dis(1, balance);
    return dis(gen);
}

void Process::log_balance() {
    logger->log_general("Current balance: " + std::to_string(current_balance.load()));
}

bool InputParser::parse_input_file(const std::string& filename, 
                                  int& n, double& lambda, double& lambda_c, int& k,
                                  std::vector<int>& initial_balances,
                                  std::map<int, std::vector<int>>& topology) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open input file: " << filename << std::endl;
         return false;
    }
    
    // Parse first line: n lambda lambda_c k
    std::string line;
    if (!std::getline(file, line)) {
        std::cerr << "Failed to read first line" << std::endl;
        return false;
    }
    
    std::istringstream iss(line);
    if (!(iss >> n >> lambda >> lambda_c >> k)) {
        std::cerr << "Failed to parse parameters" << std::endl;
        return false;
    }
    
    // Parse initial balances
    initial_balances.clear();
    int balance;
    while (static_cast<int>(initial_balances.size()) < n && file >> balance) {
        initial_balances.push_back(balance);
    }
    
    if (static_cast<int>(initial_balances.size()) != n) {
        std::cerr << "Insufficient initial balances provided" << std::endl;
        return false;
    }
    
    // Parse topology
    topology.clear();
    for (int i = 0; i < n; i++) {
        if (!std::getline(file, line)) {
            std::cerr << "Failed to read topology line " << i << std::endl;
            return false;
        }
        
        // Skip empty lines
        if (line.empty()) {
            i--; // Don't count this iteration
            continue;
        }
        
        std::istringstream line_stream(line);
        int node_id;
        if (!(line_stream >> node_id)) {
            std::cerr << "Failed to parse node ID for line " << i << std::endl;
            return false;
        }
        
        std::vector<int> neighbors;
        int neighbor;
        while (line_stream >> neighbor) {
            neighbors.push_back(neighbor);
        }
        topology[node_id] = neighbors;
    }
    
    file.close();
    return true;
}
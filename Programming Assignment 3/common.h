#ifndef COMMON_H
#define COMMON_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <thread>
#include <mutex>
#include <queue>
#include <algorithm>
#include <cmath>

// Message types
enum MessageType {
    REQUEST = 0,
    REPLY = 1,
    RELEASE = 2,
    TERM = 3
};

// Message structure
struct Message {
    MessageType type;
    int from_id;
    int timestamp;  // For Ricart-Agarwala
    int cs_number;  // Which CS entry attempt (for logging)
    int msg_count;  // For statistics
    double total_cs_wait_time; // For statistics
    
    Message() : type(REQUEST), from_id(-1), timestamp(0), cs_number(0), msg_count(0), total_cs_wait_time(0.0) {}
};

// Global configuration
struct Config {
    int n;           // number of processes
    int k;           // number of CS entries per process
    double alpha;    // average time for local computation
    double beta;     // average time inside CS
    int process_id;  // current process ID
    
    Config() : n(0), k(0), alpha(0.0), beta(0.0), process_id(0) {}
};

// Utility functions
double exponential_random(double mean);
std::string get_current_time();
void log_message(std::ofstream& logfile, const std::string& message);
int serialize_message(char* buffer, const Message& msg);
void deserialize_message(const char* buffer, Message& msg);

#endif

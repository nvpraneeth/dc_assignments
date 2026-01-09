#include "common.h"
#include <ctime>
#include <cmath>

// Generate exponential random number
double exponential_random(double mean) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::exponential_distribution<> dist(1.0 / mean);
    return dist(gen);
}

// Get current time as string
std::string get_current_time() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return ss.str();
}

// Log message to file
void log_message(std::ofstream& logfile, const std::string& message) {
    if (logfile.is_open()) {
        logfile << message << std::endl;
        logfile.flush();
    }
}

/**
 * Serialize a Message struct into a byte buffer for network transmission
 * 
 * PURPOSE:
 * TCP sockets can only send/receive raw bytes (char arrays), not C++ objects.
 * This function converts the Message struct into a flat byte array that can be
 * transmitted over the network. The struct fields are copied into the buffer in
 * a fixed order, ensuring the receiver can reconstruct the message correctly.
 * 
 * HOW IT WORKS:
 * 1. Copies each field of the Message struct into the buffer using memcpy
 * 2. Maintains a running offset to pack fields sequentially
 * 3. Returns the total number of bytes written (useful for send() call)
 * 
 * MESSAGE FORMAT (byte layout):
 * - Offset 0-3:   MessageType (4 bytes)
 * - Offset 4-7:   from_id (4 bytes, int)
 * - Offset 8-11:  timestamp (4 bytes, int)
 * - Offset 12-15: cs_number (4 bytes, int)
 * - Offset 16-19: msg_count (4 bytes, int)
 * - Offset 20-27: total_cs_wait_time (8 bytes, double)
 * Total size: 28 bytes
 * 
 * @param buffer Pointer to pre-allocated char buffer (must be at least 28 bytes)
 * @param msg Reference to the Message struct to serialize
 * @return Total number of bytes written to buffer (28 bytes)
 * 
 * USAGE:
 *   char buffer[1024];
 *   Message msg;
 *   msg.type = REQUEST;
 *   msg.from_id = 5;
 *   int size = serialize_message(buffer, msg);
 *   send(socket, buffer, size, 0);  // Send over network
 */
int serialize_message(char* buffer, const Message& msg) {
    int offset = 0;
    // Copy MessageType enum (typically 4 bytes)
    memcpy(buffer + offset, &msg.type, sizeof(MessageType));
    offset += sizeof(MessageType);
    
    // Copy process ID (4 bytes)
    memcpy(buffer + offset, &msg.from_id, sizeof(int));
    offset += sizeof(int);
    
    // Copy logical timestamp for Ricart-Agarwala (4 bytes)
    memcpy(buffer + offset, &msg.timestamp, sizeof(int));
    offset += sizeof(int);
    
    // Copy CS entry number for logging (4 bytes)
    memcpy(buffer + offset, &msg.cs_number, sizeof(int));
    offset += sizeof(int);
    
    // Copy message count for statistics (4 bytes)
    memcpy(buffer + offset, &msg.msg_count, sizeof(int));
    offset += sizeof(int);
    
    // Copy total wait time for statistics (8 bytes, double)
    memcpy(buffer + offset, &msg.total_cs_wait_time, sizeof(double));
    offset += sizeof(double);
    
    return offset;  // Return total bytes written
}

/**
 * Deserialize a byte buffer back into a Message struct
 * 
 * PURPOSE:
 * Reconstructs a Message struct from raw bytes received over the network.
 * This is the inverse operation of serialize_message(). It unpacks the byte
 * array following the same field order to recreate the original Message object.
 * 
 * HOW IT WORKS:
 * 1. Reads bytes from the buffer starting at offset 0
 * 2. Extracts each field using memcpy in the same order as serialization
 * 3. Reconstructs the complete Message struct
 * 
 * IMPORTANT:
 * - The buffer must contain at least 28 bytes of valid data
 * - Field order must match serialize_message() exactly
 * - Works correctly only if sender and receiver use same struct layout
 * 
 * @param buffer Pointer to the byte buffer received from network
 * @param msg Reference to Message struct to populate (output parameter)
 * 
 * USAGE:
 *   char buffer[1024];
 *   recv(socket, buffer, sizeof(buffer), 0);  // Receive from network
 *   Message msg;
 *   deserialize_message(buffer, msg);
 *   // Now msg contains the reconstructed message
 */
void deserialize_message(const char* buffer, Message& msg) {
    int offset = 0;
    
    // Extract MessageType enum
    memcpy(&msg.type, buffer + offset, sizeof(MessageType));
    offset += sizeof(MessageType);
    
    // Extract process ID
    memcpy(&msg.from_id, buffer + offset, sizeof(int));
    offset += sizeof(int);
    
    // Extract logical timestamp
    memcpy(&msg.timestamp, buffer + offset, sizeof(int));
    offset += sizeof(int);
    
    // Extract CS entry number
    memcpy(&msg.cs_number, buffer + offset, sizeof(int));
    offset += sizeof(int);
    
    // Extract message count
    memcpy(&msg.msg_count, buffer + offset, sizeof(int));
    offset += sizeof(int);
    
    // Extract total wait time
    memcpy(&msg.total_cs_wait_time, buffer + offset, sizeof(double));
    offset += sizeof(double);
}

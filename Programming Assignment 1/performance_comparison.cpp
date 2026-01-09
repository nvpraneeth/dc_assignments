#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <map>

using namespace std;

struct PerformanceStats {
    double avg_entries_per_message;
    int total_messages;
    int total_entries;
    double execution_time;
};

// Function to generate input file for given parameters
void generate_input_file(int n, double lambda, double alpha, int m) {
    ofstream file("inp-params.txt");
    file << n << " " << lambda << " " << alpha << " " << m << endl;
    
    // Generate ring topology (more reasonable for larger graphs)
    for (int i = 1; i <= n; i++) {
        int left = (i == 1) ? n : i - 1;
        int right = (i == n) ? 1 : i + 1;
        file << left << " " << right << endl;
    }
    file.close();
}

// Function to extract statistics from program output
PerformanceStats extract_stats(const string& output) {
    PerformanceStats stats;
    
    // Parse the output to extract statistics
    istringstream iss(output);
    string line;
    
    while (getline(iss, line)) {
        if (line.find("Average entries per message:") != string::npos) {
            size_t pos = line.find(":") + 1;
            stats.avg_entries_per_message = stod(line.substr(pos));
        }
        if (line.find("Total messages sent:") != string::npos) {
            size_t pos = line.find(":") + 1;
            stats.total_messages = stoi(line.substr(pos));
        }
        if (line.find("Total clock entries sent:") != string::npos) {
            size_t pos = line.find(":") + 1;
            stats.total_entries = stoi(line.substr(pos));
        }
    }
    
    return stats;
}

// Function to run a program and capture its output
string run_program(const string& program_name) {
    string command = "./" + program_name + " 2>&1";
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) return "";
    
    char buffer[128];
    string result = "";
    while (!feof(pipe)) {
        if (fgets(buffer, 128, pipe) != NULL)
            result += buffer;
    }
    pclose(pipe);
    return result;
}

int main() {
    cout << "Performance Comparison: Vector Clock vs Singhal-Kshemkalyani" << endl;
    cout << "==========================================================" << endl;
    
    // Parameters
    double lambda = 5.0;
    double alpha = 1.5;
    int m = 50; // Fixed number of messages per process
    
    // Results storage
    map<int, PerformanceStats> vc_results;
    map<int, PerformanceStats> sk_results;
    
    // Test with varying number of processes
    for (int n = 10; n <= 15; n++) {
        cout << "\nTesting with " << n << " processes..." << endl;
        
        // Generate input file
        generate_input_file(n, lambda, alpha, m);
        
        // Compile programs if needed
        system("g++ -std=c++17 -pthread -o VC-CS25RESCH04001 VC-CS25RESCH04001.cpp");
        system("g++ -std=c++17 -pthread -o SK-CS25RESCH04001 SK-CS25RESCH04001.cpp");
        
        // Run Vector Clock implementation
        cout << "Running Vector Clock implementation..." << endl;
        auto start = chrono::high_resolution_clock::now();
        string vc_output = run_program("VC-CS25RESCH04001");
        auto end = chrono::high_resolution_clock::now();
        
        PerformanceStats vc_stats = extract_stats(vc_output);
        vc_stats.execution_time = chrono::duration<double>(end - start).count();
        vc_results[n] = vc_stats;
        
        // Run Singhal-Kshemkalyani implementation
        cout << "Running Singhal-Kshemkalyani implementation..." << endl;
        start = chrono::high_resolution_clock::now();
        string sk_output = run_program("SK-CS25RESCH04001");
        end = chrono::high_resolution_clock::now();
        
        PerformanceStats sk_stats = extract_stats(sk_output);
        sk_stats.execution_time = chrono::duration<double>(end - start).count();
        sk_results[n] = sk_stats;
        
        // Wait a bit between runs
        this_thread::sleep_for(chrono::seconds(1));
    }
    
    // Generate results file
    ofstream results_file("performance_results.txt");
    results_file << "Performance Comparison Results" << endl;
    results_file << "=============================" << endl << endl;
    
    results_file << "Parameters:" << endl;
    results_file << "Lambda: " << lambda << endl;
    results_file << "Alpha: " << alpha << endl;
    results_file << "Messages per process: " << m << endl << endl;
    
    results_file << "Number of Processes | VC Avg Entries | SK Avg Entries | VC Messages | SK Messages | VC Time(s) | SK Time(s)" << endl;
    results_file << "-------------------|----------------|----------------|-------------|-------------|------------|-----------" << endl;
    
    for (int n = 10; n <= 15; n++) {
        results_file << n << " | " 
                    << vc_results[n].avg_entries_per_message << " | "
                    << sk_results[n].avg_entries_per_message << " | "
                    << vc_results[n].total_messages << " | "
                    << sk_results[n].total_messages << " | "
                    << vc_results[n].execution_time << " | "
                    << sk_results[n].execution_time << endl;
    }
    
    results_file << endl << "Analysis:" << endl;
    results_file << "=========" << endl;
    
    // Calculate average improvement
    double total_improvement = 0;
    int count = 0;
    
    for (int n = 10; n <= 15; n++) {
        if (vc_results[n].avg_entries_per_message > 0) {
            double improvement = (vc_results[n].avg_entries_per_message - sk_results[n].avg_entries_per_message) 
                               / vc_results[n].avg_entries_per_message * 100;
            total_improvement += improvement;
            count++;
            
            results_file << "For " << n << " processes: " << improvement << "% reduction in message size" << endl;
        }
    }
    
    if (count > 0) {
        results_file << "Average improvement: " << total_improvement / count << "%" << endl;
    }
    
    results_file.close();
    
    // Generate CSV file for plotting
    ofstream csv_file("performance_data.csv");
    csv_file << "Processes,VC_Avg_Entries,SK_Avg_Entries,VC_Messages,SK_Messages,VC_Time,SK_Time" << endl;
    
    for (int n = 10; n <= 15; n++) {
        csv_file << n << ","
                << vc_results[n].avg_entries_per_message << ","
                << sk_results[n].avg_entries_per_message << ","
                << vc_results[n].total_messages << ","
                << sk_results[n].total_messages << ","
                << vc_results[n].execution_time << ","
                << sk_results[n].execution_time << endl;
    }
    csv_file.close();
    
    cout << "\nResults saved to performance_results.txt and performance_data.csv" << endl;
    cout << "Use the CSV file to generate graphs for your report." << endl;
    
    return 0;
}

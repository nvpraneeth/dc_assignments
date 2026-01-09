# Makefile for Vector Clock and Singhal-Kshemkalyani Implementation

# Compiler settings
CXX = g++
CXXFLAGS = -std=c++17 -pthread -Wall -Wextra -O2

# Target executables
VC_TARGET = VC-CS25RESCH04001
SK_TARGET = SK-CS25RESCH04001
PERF_TARGET = performance_comparison

# Source files
VC_SRC = VC-CS25RESCH04001.cpp
SK_SRC = SK-CS25RESCH04001.cpp
PERF_SRC = performance_comparison.cpp

# Default target
all: $(VC_TARGET) $(SK_TARGET) $(PERF_TARGET)

# Compile Vector Clock implementation
$(VC_TARGET): $(VC_SRC)
	$(CXX) $(CXXFLAGS) -o $(VC_TARGET) $(VC_SRC)

# Compile Singhal-Kshemkalyani implementation
$(SK_TARGET): $(SK_SRC)
	$(CXX) $(CXXFLAGS) -o $(SK_TARGET) $(SK_SRC)

# Compile performance comparison
$(PERF_TARGET): $(PERF_SRC)
	$(CXX) $(CXXFLAGS) -o $(PERF_TARGET) $(PERF_SRC)

# Run Vector Clock
run-vc: $(VC_TARGET)
	./$(VC_TARGET)

# Run Singhal-Kshemkalyani
run-sk: $(SK_TARGET)
	./$(SK_TARGET)

# Run performance comparison
run-perf: $(PERF_TARGET)
	./$(PERF_TARGET)

# Generate graphs (requires Python)
graphs: run-perf
	python3 generate_graphs.py

# Clean build files
clean:
	rm -f $(VC_TARGET) $(SK_TARGET) $(PERF_TARGET)
	rm -f *.log *.txt *.csv *.png

# Install Python dependencies
install-deps:
	pip3 install pandas matplotlib seaborn numpy

# Full test run
test: all run-vc run-sk run-perf graphs

# Help target
help:
	@echo "Available targets:"
	@echo "  all        - Compile all executables"
	@echo "  run-vc     - Run Vector Clock implementation"
	@echo "  run-sk     - Run Singhal-Kshemkalyani implementation"
	@echo "  run-perf   - Run performance comparison"
	@echo "  graphs     - Generate performance graphs"
	@echo "  install-deps - Install Python dependencies"
	@echo "  test       - Run complete test suite"
	@echo "  clean      - Remove compiled files"
	@echo "  help       - Show this help message"

.PHONY: all run-vc run-sk run-perf graphs install-deps test clean help

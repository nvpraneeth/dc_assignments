CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra -O2 -pthread
TARGET = banking_system
TARGET_PARALLEL = main_parallel
SOURCES = process.cpp message.cpp
OBJECTS = $(SOURCES:.cpp=.o)

all: generate-inputs $(TARGET) $(TARGET_PARALLEL)

$(TARGET): main.o $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) main.o $(OBJECTS)

$(TARGET_PARALLEL): main_parallel.o $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET_PARALLEL) main_parallel.o $(OBJECTS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o $(TARGET) $(TARGET_PARALLEL)
	rm -rf logs/ results/
	rm -f exp_*.txt inp-params-*.txt

run: generate-inputs $(TARGET)
	./$(TARGET) inp-params-line-3.txt

run-parallel: generate-inputs $(TARGET_PARALLEL)
	./$(TARGET_PARALLEL) inp-params-line-3.txt

experiment: $(TARGET)
	./$(TARGET) inp-params.txt experiment

parallel-experiments: $(TARGET_PARALLEL)
	chmod +x run_parallel_experiments.sh
	chmod +x generate_inputs.py
	./run_parallel_experiments.sh

generate-inputs:
	python3 generate_inputs.py

test-topologies: $(TARGET_PARALLEL)
	@echo "Testing individual topologies..."
	@for file in inp-params-*.txt; do \
		if [ -f "$$file" ]; then \
			echo "Testing $$file"; \
			./$(TARGET_PARALLEL) $$file; \
		fi; \
	done

.PHONY: all clean run run-parallel experiment parallel-experiments generate-inputs test-topologies

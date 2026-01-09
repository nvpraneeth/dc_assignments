# Makefile for Distributed Computing Programming Assignment 3
# Maekawa's and Ricart-Agarwala's Algorithms

CXX = g++
CXXFLAGS = -std=c++11 -pthread -Wall -O2
LDFLAGS = -pthread

# Source files
COMMON_SRC = common.cpp
MK_SRC = MK-CS25RESCH04001.cpp
RA_SRC = RA-CS25RESCH04001.cpp

# Object files
COMMON_OBJ = common.o
MK_OBJ = MK-CS25RESCH04001.o
RA_OBJ = RA-CS25RESCH04001.o

# Executables
MK_EXEC = mk-node
RA_EXEC = ra-node

.PHONY: all mk ra clean

all: mk ra

mk: $(MK_EXEC)

ra: $(RA_EXEC)

$(MK_EXEC): $(MK_OBJ) $(COMMON_OBJ)
	$(CXX) $(CXXFLAGS) -o $(MK_EXEC) $(MK_OBJ) $(COMMON_OBJ) $(LDFLAGS)

$(RA_EXEC): $(RA_OBJ) $(COMMON_OBJ)
	$(CXX) $(CXXFLAGS) -o $(RA_EXEC) $(RA_OBJ) $(COMMON_OBJ) $(LDFLAGS)

$(MK_OBJ): $(MK_SRC) common.h
	$(CXX) $(CXXFLAGS) -c $(MK_SRC) -o $(MK_OBJ)

$(RA_OBJ): $(RA_SRC) common.h
	$(CXX) $(CXXFLAGS) -c $(RA_SRC) -o $(RA_OBJ)

$(COMMON_OBJ): $(COMMON_SRC) common.h
	$(CXX) $(CXXFLAGS) -c $(COMMON_SRC) -o $(COMMON_OBJ)

clean:
	rm -f $(MK_EXEC) $(RA_EXEC) $(MK_OBJ) $(RA_OBJ) $(COMMON_OBJ)
	rm -f *.log mk_stats.txt ra_stats.txt

PROGRAM := main

CXX := g++

EIGENDIR :=

CXXFLAGS := -std=c++20 -Ofast -Wall -g -I$(EIGENDIR) -fopenmp


SRCS := main.cpp instance.cpp node.cpp beam_search.cpp nnet.cpp

OBJS := $(SRCS:.cpp=.o)

all: $(PROGRAM)

$(PROGRAM): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

.PHONY: clean
clean:
	$(RM) $(OBJS) $(PROGRAM)


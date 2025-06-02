#pragma once

#include <vector>
#include <map>
#include <tuple>

class Instance;

using rlcs_position = std::tuple<std::vector<int>, std::vector<int>, std::vector<int>>;

class Node {
public:
    Instance* inst;
    rlcs_position position; // tuple of (S positions, P positions, R positions)
    Node* parent = nullptr; // pointer to parent node

    int l_v = 0; // depth level
    int f_value = 0;
    double heuristic_value = 0.0; // heuristic value
    bool complete = false; 

    std::vector<double> features; // features for the NN

public:

Node(Instance* instance, std::vector<int>& left,
         std::vector<int>& ppos, std::vector<int>& rpos,
         Node* parent = nullptr);

    
    std::vector<Node*> expansion();               
    std::vector<int> export_solution();           
    bool is_complete();                             

    bool domination_two_letters(rlcs_position& posA, rlcs_position& posB);
    std::map<int, rlcs_position> sigma_feasible_letters();

    double greedy_function();                    
    void set_up_prob_value(int k);                

    void print(); // debugging

    bool operator>(const Node* other);
};


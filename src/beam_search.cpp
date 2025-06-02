#include <Eigen/Dense>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include <cmath>
#include <set>
#include <chrono>
#include <algorithm>
#include "beam_search.h"
#include "nnet.h"
#include "node.h"
#include "instance.h"

using namespace std;

double running_time = 0.0;
std::vector<int> solution;

bool validate_solution(Instance* inst) {
    if (solution.empty()) return false;

    // input strings: solution must be a subsequence of each S[i]
    for (int i = 0; i < inst->m; ++i) {
        size_t count = 0;
        for (auto& si : inst->S[i]) {
            if (si == solution[count]) count++;
            if (count == solution.size()) break;
        }
        if (count < solution.size()) return false;
    }

    // P-strings: must be subsequence of the solution
    for (int j = 0; j < inst->p; ++j) {
        
size_t count = 0;
        for (auto& s : solution) {
            if (inst->P[j][count] == s) count++;
            if (count == inst->P[j].size()) break;
        }
        if (count < inst->P[j].size()) return false;
    }

    // R-strings: must not be a subsequence of the solution
    for (int k = 0; k < inst->r; ++k) {
        size_t count = 0;
        for (auto& s : solution) {
            if (s == inst->R[k][count]) count++;
            if (count == inst->R[k].size()) return false;
        }
    }

    return true;
}

void save_in_file(const std::string& outfile, Instance* inst) {
    // Clean file name
    string clean_file_name;
    size_t bar_count = count(inst->file_name.begin(), inst->file_name.end(), '/');
    size_t count = 0;
    for (char c : inst->file_name) {
        if (count == bar_count && c == '.') break;
        if (count == bar_count) clean_file_name.push_back(c);
        else if (c == '/') count++;
    }

    if (outfile.empty()) {
        std::cout << clean_file_name << std::endl;
        std::cout << "Objective: " << solution.size() << std::endl;
        std::cout << "Solution: ";
        for (auto& c : solution) std::cout << inst->int2char[c] << " ";
        std::cout << "\nTime: " << running_time << std::endl;
        std::cout << "Feasible: " << validate_solution(inst) << std::endl;
        return;
    }
    // Save to file
    std::ofstream outputFile(outfile);
    if (!outputFile) {
        std::cerr << "Error opening the file." << std::endl;
        return;
    }

    outputFile << clean_file_name << std::endl;
    outputFile << "Objective: " << solution.size() << std::endl;
    outputFile << "Solution: ";
    for (auto& c : solution) outputFile << inst->int2char[c] << " ";
    outputFile << "\nTime: " << running_time << std::endl;
    outputFile << "Feasible: " << validate_solution(inst) << std::endl;
}

double compute_max(const vector<double>& values) {
    return *max_element(values.begin(), values.end());
}

double compute_min(const vector<double>& values) {
    return *min_element(values.begin(), values.end());
}

double compute_average(const vector<double>& values) {
    return accumulate(values.begin(), values.end(), 0.0) / values.size();
}

double compute_std(const vector<double>& values, double mean) {
    double sum_sq = 0.0;
    for (double v : values)
        sum_sq += (v - mean) * (v - mean);
    return sqrt(sum_sq / values.size());
}

void standardize(vector<double>& features) {
    double avg = compute_average(features);
    double std_dev = compute_std(features, avg);
    for (double& f : features)
        f = (f - avg) / std_dev;
}

void compute_features(vector<Node*>& V_ext, int feature_config) {
    for (Node* node : V_ext) {
        vector<double> pL_v(get<0>(node->position).begin(), get<0>(node->position).end());
        vector<double> lv(get<2>(node->position).begin(), get<2>(node->position).end());

        // normalize left position vectors with respect to input and restricted strings lengths
        for (size_t i = 0; i < pL_v.size(); ++i) pL_v[i] /= node->inst->S[i].size();
        for (size_t i = 0; i < lv.size(); ++i) lv[i] /= node->inst->R[i].size();

        vector<double> features = {
            compute_max(pL_v), compute_min(pL_v), compute_average(pL_v), compute_std(pL_v, compute_average(pL_v)),
            compute_max(lv), compute_min(lv), compute_average(lv), compute_std(lv, compute_average(lv)),
            static_cast<double>(node->l_v)
        };

        if (feature_config >= 2) features.push_back(node->inst->Sigma);
        if (feature_config >= 3) {
            features.push_back(node->inst->m);
            features.push_back(node->inst->r);
        }
        if (feature_config == 4) {
            features.push_back(node->inst->S[0].size()); // assumes uniform input length
            features.push_back(node->inst->R[0].size());
        }

        standardize(features);
        node->features = features;
    }
}

void compute_heuristic_values(vector<Node*>& V_ext, MLP& neural_network) {
    for (Node* node : V_ext) {
        Eigen::Map<const Eigen::VectorXd> eigen_features(node->features.data(), node->features.size());
        node->heuristic_value = neural_network.forward(eigen_features)(0);
    }
}

double BS(double t_lim, int beta, Instance* inst, MLP& neural_network, bool training) {
    std::vector<int> pL(inst->m, 0), ppl(inst->p, 0), rpl(inst->r, 0);
    Node* root = new Node(inst, pL, ppl, rpl);

    vector<Node*> beam = {root}, to_delete = {root};
    Node* best_node = root;
    int l_best = 0;

    auto start_time = chrono::high_resolution_clock::now();
    set<tuple<vector<int>, vector<int>>> seen_nodes;

    while (!beam.empty()) {
        vector<Node*> V_ext;

        for (Node* node : beam) {
            auto children = node->expansion();
            if (children.empty() && node->l_v > l_best && node->is_complete()) {
                l_best = node->l_v;
                best_node = node;
            }

            for (Node* child : children) {
                auto key = make_tuple(get<0>(child->position), get<2>(child->position));
                if (seen_nodes.insert(key).second) {
                    V_ext.push_back(child);
                } else {
                    delete child;
                }
            }
        }

        compute_features(V_ext, neural_network.feature_config);
        compute_heuristic_values(V_ext, neural_network);

        sort(V_ext.begin(), V_ext.end(),
            [](Node* a, Node* b) { return a->heuristic_value > b->heuristic_value; });

        beam.clear();
        for (size_t i = 0; i < min(V_ext.size(), static_cast<size_t>(beta)); ++i) {
            beam.push_back(V_ext[i]);
            to_delete.push_back(V_ext[i]);
        }

        for (size_t i = beta; i < V_ext.size(); ++i)
            delete V_ext[i];

        seen_nodes.clear();

        auto duration = chrono::duration_cast<chrono::milliseconds>(
                            chrono::high_resolution_clock::now() - start_time);
        running_time = duration.count() / 1000.0;

        if (running_time >= t_lim) break;
    }

    if (!training) {
        solution = best_node->export_solution();
        save_in_file(neural_network.output_filename, inst);
    }

    for (Node* node : to_delete) delete node;
    return l_best;
}

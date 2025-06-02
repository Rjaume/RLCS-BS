#include <iostream>
#include <tuple>
#include <algorithm>
#include <map>
#include <vector>
#include <set>
#include "node.h"
#include "instance.h"

Node::Node(Instance* instance, std::vector<int>& left, std::vector<int>& ppos, std::vector<int>& rpos, Node* parent)
    : inst(instance), parent(parent) {
    
    position = std::make_tuple(left, ppos, rpos);
    l_v = (parent == nullptr) ? 0 : parent->l_v + 1;
    f_value = l_v + heuristic_value;
}

bool Node::domination_two_letters(rlcs_position& posA, rlcs_position& posB) {
    for (int i = 0; i < inst->m; ++i)
        if (std::get<0>(posA)[i] < std::get<0>(posB)[i])
            return false;

    for (int j = 0; j < inst->p; ++j)
        if (std::get<1>(posA)[j] > std::get<1>(posB)[j])
            return false;

    for (int i = 0; i < inst->r; ++i)
        if (std::get<2>(posA)[i] <= std::get<2>(posB)[i])
            return false;

    return true;
}

std::map<int, rlcs_position> Node::sigma_feasible_letters() {
    std::vector<int> sigma;
    for (int l = 0; l < inst->Sigma; ++l) {
        bool feasible = true;
        for (int i = 0; i < inst->m && feasible; ++i) {
            if (std::get<0>(position)[i] >= (int)inst->S[i].size())
                feasible = false;
            else if (inst->occurances_string_pos_char[l][i][std::get<0>(position)[i]] <= 0)
                feasible = false;
        }
        if (feasible)
            sigma.push_back(l);
    }

    std::map<int, rlcs_position> letters_next_positions;
    std::vector<int> letters_to_remove;

    for (int lett : sigma) {
        std::vector<int> pl_next, pleft_next, rleft_next;

        for (int i = 0; i < inst->m; ++i) {
            int pl_i = inst->next_char_occurance_in_strings[lett][i][std::get<0>(position)[i]] + 1;
            pl_next.push_back(pl_i);
        }

        for (int j = 0; j < (int)inst->P.size(); ++j) {
            int pj_left = std::get<1>(position)[j];
            pleft_next.push_back(((int)inst->P[j].size() > pj_left && inst->P[j][pj_left] == lett) ? pj_left + 1 : pj_left);
        }

        for (int k = 0; k < (int)inst->R.size(); ++k) {
            int rk_left = std::get<2>(position)[k];
            rleft_next.push_back((inst->R[k][rk_left] == lett) ? rk_left + 1 : rk_left);
            if (rleft_next[k] >= (int)inst->R[k].size())
                letters_to_remove.push_back(lett);
        }

        letters_next_positions.emplace(lett, std::make_tuple(pl_next, pleft_next, rleft_next));
    }

    // embed structure check
    for (auto& it : letters_next_positions) {
        if (std::find(letters_to_remove.begin(), letters_to_remove.end(), it.first) != letters_to_remove.end())
            continue;

        auto& pl_left = std::get<0>(it.second);
        auto& pleft = std::get<1>(it.second);

        bool feasible = true;
        for (int j = 0; j < (int)pleft.size() && feasible; ++j) {
            for (int i = 0; i < inst->m && feasible; ++i) {
                if (pl_left[j] < (int)inst->P[j].size() &&
                    inst->remaining_patern_suffix_pos[i][j][pleft[j]] < pl_left[i]) {
                    feasible = false;
                }
            }
        }

        if (!feasible)
            letters_to_remove.push_back(it.first);
    }

    for (int letter : letters_to_remove)
        letters_next_positions.erase(letter);

    // domination pruning
    for (auto itA = letters_next_positions.begin(); itA != letters_next_positions.end(); ++itA) {
        for (auto itB = letters_next_positions.begin(); itB != letters_next_positions.end(); ++itB) {
            if (itA != itB && domination_two_letters(itA->second, itB->second))
                letters_to_remove.push_back(itA->first);
        }
    }

    for (int letter : letters_to_remove)
        letters_next_positions.erase(letter);

    return letters_next_positions;
}

std::vector<Node*> Node::expansion() {
    auto letters_next_positions = sigma_feasible_letters();
    std::vector<Node*> node_extensions;

    for (auto& [lett, pos] : letters_next_positions) {
        node_extensions.push_back(new Node(inst, std::get<0>(pos), std::get<1>(pos), std::get<2>(pos), this));
    }

    return node_extensions;
}

bool Node::is_complete() {
    auto& lleft = std::get<1>(position);
    auto& rleft = std::get<2>(position);

    for (int i = 0; i < (int)lleft.size(); ++i)
        if (lleft[i] < (int)inst->P[i].size())
            return false;

    for (int k = 0; k < (int)rleft.size(); ++k)
        if (rleft[k] >= (int)inst->R[k].size())
            return false;

    return true;
}

std::vector<int> Node::export_solution() {
    std::vector<int> solution;
    int next_in_s0 = std::get<0>(position)[0];

    if (next_in_s0 == 0)
        return solution;

    int add_letter = inst->S[0][next_in_s0 - 1];
    Node* node = parent;

    if (node->l_v + 1 == l_v)
        solution.push_back(add_letter);

    next_in_s0 = std::get<0>(node->position)[0];

    while (next_in_s0 > 0) {
        add_letter = node->inst->S[0][next_in_s0 - 1];
        int l_v_child = node->l_v;
        node = node->parent;
        next_in_s0 = std::get<0>(node->position)[0];

        if (node->l_v + 1 == l_v_child)
            solution.push_back(add_letter);
    }

    std::reverse(solution.begin(), solution.end());
    return solution;
}

bool Node::operator>(const Node* other) {
    return (f_value == other->f_value) ? (l_v > other->l_v) : (f_value > other->f_value);
}


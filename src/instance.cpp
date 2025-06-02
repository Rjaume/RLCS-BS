#include "instance.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cassert>
#include <algorithm> 
#include <iterator>  

Instance::Instance(const std::string& path)
{
    file_name = path;

    std::ifstream inputFile(path);
    if (!inputFile.is_open()) {
        std::cerr << "Error opening the file: " << path << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string line;
    bool first_line = true;
    std::vector<std::string> strings;
    int num_map = 0;

    while (std::getline(inputFile, line)) {
        std::istringstream ss(line);

        if (first_line) {
            int m, sigma, p, r;
            if (ss >> m >> sigma >> p >> r) {
                this->m = m;
                this->Sigma = sigma;
                this->p = p;
                this->r = r;
            } else {
                std::cerr << "Error parsing first line: " << line << " of file " << path << std::endl;
                exit(EXIT_FAILURE);
            }
            first_line = false;
        } else {
            int length;
            std::string si;
            if (ss >> length >> si) {
                for (char c : si) {
                    if (map_char_to_int.find(c) == map_char_to_int.end()) {
                        map_char_to_int[c] = num_map++;
                    }
                }
                strings.push_back(si);
            } else {
                std::cerr << "Error parsing line: " << line << " of file " << path << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    }
    inputFile.close();

    auto convert_string_to_int_vector = [this](const std::string& s) {
        std::vector<int> vec;
        vec.reserve(s.size());
        for (char c : s) {
            assert(c >= 0);
            vec.push_back(map_char_to_int[c]);
        }
        return vec;
    };

    for (int i = 0; i < m; ++i) {
        S.push_back(convert_string_to_int_vector(strings[i]));
    }
    for (int i = m; i < m + p; ++i) {
        P.push_back(convert_string_to_int_vector(strings[i]));
    }
    for (int i = m + p; i < m + p + r; ++i) {
        R.push_back(convert_string_to_int_vector(strings[i]));
    }

    for (const auto& [key, val] : map_char_to_int) {
        int2char[val] = key;
    }

    fill_in_data_structures();
}

void Instance::fill_in_data_structures()
{
    // occurances_string_pos_char
    occurances_string_pos_char.resize(Sigma);
    for (int a = 0; a < Sigma; ++a) {
        std::vector<std::vector<int>> occur_a_all_s;
        occur_a_all_s.reserve(m);

        for (int i = 0; i < m; ++i) {
            const auto& s = S[i];
            std::vector<int> pos_occur_a_in_si(s.size());
            int count = 0;

            for (int j = (int)s.size() - 1; j >= 0; --j) {
                if (s[j] == a)
                    ++count;
                pos_occur_a_in_si[j] = count;
            }
            occur_a_all_s.push_back(std::move(pos_occur_a_in_si));
        }
        occurances_string_pos_char[a] = std::move(occur_a_all_s);
    }

    // next_char_occurance_in_strings
    next_char_occurance_in_strings.resize(Sigma);
    for (int a = 0; a < Sigma; ++a) {
        std::vector<std::vector<int>> next_occur_a_all_s;
        next_occur_a_all_s.reserve(m);

        for (int i = 0; i < m; ++i) {
            const auto& s = S[i];
            int next_pos = (int)s.size();
            std::vector<int> next_occur(s.size(), next_pos);

            for (int j = (int)s.size() - 1; j >= 0; --j) {
                if (s[j] == a)
                    next_pos = j;
                next_occur[j] = next_pos;
            }
            next_occur_a_all_s.push_back(std::move(next_occur));
        }
        next_char_occurance_in_strings[a] = std::move(next_occur_a_all_s);
    }

    // remaining_patern_suffix_pos
    remaining_patern_suffix_pos.resize(m);
    for (int i = 0; i < m; ++i) {
        std::vector<std::vector<int>> embedding_all_pj_into_si;
        embedding_all_pj_into_si.reserve(p);

        for (int j = 0; j < p; ++j) {
            const auto& pj = P[j];
            std::vector<int> embedding_pj_into_si(pj.size(), -1);
            int max_index = (int)pj.size() - 1;

            const auto& si = S[i];
            for (int its = (int)si.size() - 1; its >= 0 && max_index >= 0; --its) {
                if (pj[max_index] == si[its]) {
                    embedding_pj_into_si[max_index] = its;
                    --max_index;
                }
            }
            embedding_all_pj_into_si.push_back(std::move(embedding_pj_into_si));
        }
        remaining_patern_suffix_pos[i] = std::move(embedding_all_pj_into_si);
    }
}

Instance::~Instance()
{
    S.clear();
    P.clear();
    R.clear();
    map_char_to_int.clear();
    int2char.clear();
}


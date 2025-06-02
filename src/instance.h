#pragma once

#include <vector>
#include <map>
#include <string>

class Instance
{
public:
    std::string file_name;
    int m = 0;           // number of S-strings
    int Sigma = 0;       // alphabet size
    int p = 0;           // number of P-strings
    int r = 0;           // number of R-strings

    std::vector<std::vector<int>> S; // S-strings encoded as int vectors
    std::vector<std::vector<int>> P; // P-strings encoded as int vectors
    std::vector<std::vector<int>> R; // R-strings encoded as int vectors

    std::map<char, int> map_char_to_int;  // char -> int mapping for alphabet
    std::map<int, char> int2char;         // reverse mapping

    //vectors for various preprocessed info:
    // occurances_string_pos_char[char][i][j] = number of occurrences of char in S[i][j..end]
    std::vector<std::vector<std::vector<int>>> occurances_string_pos_char;

    // next_char_occurance_in_strings[char][i][j] = position of next occurrence of char in S[i] at or after index j
    std::vector<std::vector<std::vector<int>>> next_char_occurance_in_strings;

    // remaining_patern_suffix_pos[i][j][px] = max index of S[i] where P[j][px..end] can be embedded
    std::vector<std::vector<std::vector<int>>> remaining_patern_suffix_pos;

public:
    explicit Instance(const std::string& path);
    ~Instance();

    void fill_in_data_structures();
};

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>
#include <cstring>
#include "beam_search.h"
#include "nnet.h"
#include "instance.h"

#ifdef _OPENMP
    #include <omp.h>
#else
    #pragma message("Warning: Compiling without OpenMP support.")
#endif

constexpr int MAX_THREADS = 20;

MLP neural_network;
bool training = true;
bool parallel = false;
int beam_width;
double time_limit;
int hidden_layers;
int num_threads;
int num_features;
std::string filename;

std::vector<int> units;
std::vector<std::string> training_files;
std::vector<std::string> validation_files;

void read_parameters(int argc, char** argv) {
    int i = 1;
    while (i < argc) {
        std::string arg = argv[i];
        if (arg == "-weight_limit") neural_network.weight_limit = std::stoi(argv[++i]);
        else if (arg == "-training_beam_width") neural_network.training_beam_width = std::stoi(argv[++i]);
        else if (arg == "-training_time_limit") neural_network.training_time_limit = std::stod(argv[++i]);
        else if (arg == "-hidden_layers") hidden_layers = std::stoi(argv[++i]);
        else if (arg == "-units") {
            for (int j = 0; j < hidden_layers; ++j)
                units.push_back(std::stoi(argv[++i]));
        }
        else if (arg == "-time_limit") time_limit = std::stod(argv[++i]);
        else if (arg == "-i") { training = false; filename = argv[++i]; }
        else if (arg == "-o") neural_network.output_filename = argv[++i];
        else if (arg == "-beam_width") beam_width = std::stoi(argv[++i]);
        else if (arg == "-activation_function") neural_network.activation_function = std::stoi(argv[++i]);
        else if (arg == "-feature_configuration") {
            neural_network.feature_config = std::stoi(argv[++i]);
            switch (neural_network.feature_config) {
                case 1: num_features = 9; break;
                case 2: num_features = 10; break;
                case 3: num_features = 12; break;
                case 4: num_features = 14; break;
                default: num_features = 9;
            }
        }
        else if (arg == "-ga_configuration") neural_network.ga_config = std::stoi(argv[++i]);
        else if (arg == "-population_size") neural_network.population_size = std::stoi(argv[++i]);
        else if (arg == "-n_elites") neural_network.n_elites = std::stoi(argv[++i]);
        else if (arg == "-n_mutants") neural_network.n_mutants = std::stoi(argv[++i]);
        else if (arg == "-rho") neural_network.elite_inheritance_probability = std::stod(argv[++i]);
        else if (arg == "-parallel") parallel = true;
        else if (arg == "-num_threads") num_threads = std::stoi(argv[++i]);
        ++i;
    }

    if (!training) return;

    std::ifstream path_file("instances_path.txt");
    std::string base_path;
    if (!path_file || !std::getline(path_file, base_path)) {
        std::cerr << "Error: Missing or unreadable 'instances_path.txt' file.\n";
        exit(EXIT_FAILURE);
    }

    for (const std::string file_name : {"training_instances.txt", "validation_instances.txt"}) {
        std::ifstream file(file_name);
        if (!file) {
            std::cerr << "Error: Could not open '" << file_name << "'.\n";
            exit(EXIT_FAILURE);
        }

        std::string instance;
        auto& target = (file_name == std::string("training_instances.txt")) ? training_files : validation_files;
        while (std::getline(file, instance)) {
            target.push_back(base_path + instance);
        }
    }
}

void set_up_neural_network() {
    // Define architecture
    neural_network.units_per_layer.push_back(num_features);
    for (int u : units) neural_network.units_per_layer.push_back(u);
    neural_network.units_per_layer.push_back(1); // output

    if (training) {
        if (parallel) {
            if (num_threads == 0) {
                if (training_files.size() <= MAX_THREADS) {
                    num_threads = static_cast<int>(training_files.size());
                } else {
                    std::cerr << "Error: More training files than threads available. Use -num_threads.\n";
                    exit(EXIT_FAILURE);
                }
            }

            #ifdef _OPENMP
                omp_set_num_threads(num_threads);
                std::cerr << "Training in parallel with " << num_threads << " threads.\n";
            #else
                std::cerr << "Error: OpenMP not enabled but parallel mode requested.\n";
                exit(EXIT_FAILURE);
            #endif
        } else {
            #ifdef _OPENMP
                omp_set_num_threads(1);
            #endif
            std::cerr << "Training in single-threaded mode.\n";
        }

        neural_network.training_instances.reserve(training_files.size());
        neural_network.validation_instances.reserve(validation_files.size());
        for (const auto& file : training_files) neural_network.training_instances.emplace_back(Instance(file));
        for (const auto& file : validation_files) neural_network.validation_instances.emplace_back(Instance(file));

    } else {
        std::ifstream weights_in("weights.txt");
        if (!weights_in) {
            std::cerr << "Error: Could not open weights file.\n";
            exit(EXIT_FAILURE);
        }
        std::vector<double> weights;
        double w;
        while (weights_in >> w) weights.push_back(w);
        neural_network.store_weights(weights);
    }

    if (neural_network.activation_function < 1 || neural_network.activation_function > 3)
        std::cerr << "Warning: No activation function set. Use -activation_function {1: tanh, 2: relu, 3: sigmoid}.\n";

    if (neural_network.ga_config < 1 || neural_network.ga_config > 3)
        std::cerr << "Warning: GA configuration not specified. Defaulting to 1 (rkga).\n";

    if (neural_network.weight_limit == 0){
        std::cerr << "Warning: Weight limit not set. Defaulting to 1.";
        neural_network.weight_limit = 1;
    }

    if (neural_network.units_per_layer.empty()) {
        std::cerr << "Error: Neural network architecture not properly defined.\n";
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv) {
    read_parameters(argc, argv);

    set_up_neural_network();

    std::cout << std::setprecision(10) << std::fixed;
    if (training) {
        std::vector<double> final_weights = neural_network.Train();

        std::ofstream weights_out("last_weights.txt");
        for (double w : final_weights)
            weights_out << w << " ";
    } else {
        auto* instance = new Instance(filename);
        BS(time_limit, beam_width, instance, neural_network, false);
        delete instance;
    }

    return 0;
}

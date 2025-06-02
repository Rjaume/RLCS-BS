#pragma once

#include <vector>
#include <fstream>
#include <iostream>
#include <Eigen/Dense>

class Instance;

struct training_individual {
    std::vector<double> weights;  // chromosome: a set of neural network weights
    double ofv;                   // objective function value (quality of weights)
};

class MLP {
public:
    // architecture and learned parameters
    std::vector<size_t> units_per_layer;
    std::vector<Eigen::MatrixXd> bias_vectors;
    std::vector<Eigen::MatrixXd> weight_matrices;
    std::vector<Eigen::MatrixXd> activations;

    // training and validation data
    std::vector<Instance> training_instances;
    std::vector<Instance> validation_instances;

    // general training configuration
    std::string output_filename;
    int training_beam_width = 0;
    double training_time_limit = 0.0;
    double weight_limit = 1.0;

    // feature/activation configuration
    int activation_function = 0;
    int feature_config = 1;

    // GA configuration
    int ga_config = 1;
    int population_size = 20;
    int n_elites = 1;
    int n_mutants = 7;
    double elite_inheritance_probability = 0.5;

    MLP();

    Eigen::VectorXd forward(const Eigen::VectorXd& x);
    void apply_activation_function(Eigen::MatrixXd& x);

    void store_weights(const std::vector<double>& weights);
    double calculate_validation_value(const std::vector<double>& weights);

    std::vector<double> Train();
    void apply_decoder(training_individual& ind);

    void write_weights_to_file(const std::vector<double>& weights, double time);
    void write_training_and_validation_values(std::ofstream& training_values_file,
                                              std::ofstream& validation_values_file,
                                              double time,
                                              int niter,
                                              double training_value,
                                              double validation_value);
};


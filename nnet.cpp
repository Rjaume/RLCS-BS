#include "nnet.h"
#include "beam_search.h"
#include "instance.h"

#include <random>
#include <numeric>
#include <algorithm>
#include <chrono>
#ifdef _OPENMP
    #include <omp.h>
#endif

constexpr int training_bs_time_limit = 10;

// random engine setup
unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
std::default_random_engine generator(seed);
std::uniform_real_distribution<double> standard_distribution_01(0.0, 1.0);

// activation functions
Eigen::MatrixXd relu(const Eigen::MatrixXd& x) {
    return x.array().max(0);
}

Eigen::MatrixXd sigmoid(const Eigen::MatrixXd& x) {
    return 1.0 / (1.0 + (-x).array().exp());
}

void MLP::apply_activation_function(Eigen::MatrixXd& x) {
    if (activation_function == 1) {
        x = x.array().tanh();
    } else if (activation_function == 2) {
        x = relu(x);
    } else if (activation_function == 3) {
        x = sigmoid(x);
    }
}

int produce_random_integer(int max, double rval) {
    int num = static_cast<int>(double(max) * rval);
    return (num == max) ? num - 1 : num;
}

void print_information(double best_ofv, double ctime, int niter, double validation_value) {
    std::cout << std::endl << "-----------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "best: " << best_ofv << " | time: " << ctime << " | iteration: " << niter + 1 
              << " | validation value: " << validation_value << std::endl;
    std::cout << "-----------------------------------------------------------------------------------------------------" << std::endl;
}

MLP::MLP() {}

Eigen::VectorXd MLP::forward(const Eigen::VectorXd& x) {
    Eigen::MatrixXd prev = x;
    for (size_t i = 0; i < units_per_layer.size() - 1; ++i) {
        Eigen::MatrixXd y = weight_matrices[i] * prev + bias_vectors[i];
        apply_activation_function(y);
        prev = y;
    }
    return prev;
}

void MLP::write_weights_to_file(const std::vector<double>& weights, double time) {
    std::ofstream weights_file("weights_" + std::to_string(time) + ".txt");
    for (double weight : weights)
        weights_file << weight << " ";
}

double MLP::calculate_validation_value(const std::vector<double>& weights) {
    store_weights(weights);
    double validation_value = 0.0;
    #pragma omp parallel for reduction(+:validation_value)
    for (size_t i = 0; i < validation_instances.size(); ++i) {
        validation_value += BS(training_bs_time_limit, training_beam_width, &validation_instances[i], *this, true);
    }
    return validation_value / validation_instances.size();
}

void MLP::apply_decoder(training_individual& ind) {
    store_weights(ind.weights);
    double ofv = 0.0;
    #pragma omp parallel for reduction(+:ofv)
    for (size_t i = 0; i < training_instances.size(); ++i) {
        ofv += BS(training_bs_time_limit, training_beam_width, &training_instances[i], *this, true);
    }
    ind.ofv = ofv / training_instances.size();
}

void MLP::store_weights(const std::vector<double>& weights) {
    weight_matrices.clear();
    bias_vectors.clear();

    int idx = 0;
    for (size_t i = 0; i < units_per_layer.size() - 1; ++i) {
        int rows = units_per_layer[i + 1];
        int cols = units_per_layer[i];

        Eigen::MatrixXd w(rows, cols);
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c)
                w(r, c) = weights[idx++];

        weight_matrices.push_back(w);

        Eigen::MatrixXd b(rows, 1);
        for (int r = 0; r < rows; ++r)
            b(r) = weights[idx++];

        bias_vectors.push_back(b);
    }
}

void MLP::write_training_and_validation_values(std::ofstream& train_file,
                                               std::ofstream& valid_file,
                                               double time,
                                               int niter,
                                               double train_val,
                                               double valid_val) {
    train_file << time << "\t" << niter << "\t" << train_val << std::endl;
    valid_file << time << "\t" << niter << "\t" << valid_val << std::endl;
}

std::vector<double> MLP::Train() {
    std::uniform_real_distribution<double> weight_dist(-weight_limit, weight_limit);
    
    int n_weights = 0;
    for (size_t i = 0; i < units_per_layer.size() - 1; ++i)
        n_weights += (units_per_layer[i] + 1) * units_per_layer[i + 1];

    std::ofstream training_file("training_values.txt");
    std::ofstream validation_file("validation_values.txt");
    training_file << "Time\tGenerations\tTraining value" <<std::endl;
    validation_file << "Time\tGenerations\tValidation value" << std::endl;

    auto start = std::chrono::steady_clock::now();
    bool stop = false;
    double ctime = 0.0;
    int niter = 0;

    int n_offspring = population_size - n_elites - n_mutants;

    std::vector<training_individual> population(population_size);
    std::vector<double> best_weights;
    double best_ofv = std::numeric_limits<double>::lowest();

    // initialize population
    for (int pi = 0; pi < population_size && !stop; ++pi) {
        population[pi].weights.resize(n_weights);
        for (double& w : population[pi].weights)
            w = weight_dist(generator);
        
        apply_decoder(population[pi]);

        ctime = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (ctime > training_time_limit) stop = true;

        if (population[pi].ofv > best_ofv) {
            best_ofv = population[pi].ofv;
            best_weights = population[pi].weights;
            write_weights_to_file(best_weights, ctime);
            double validation_value = calculate_validation_value(best_weights);
            write_training_and_validation_values(training_file, validation_file, ctime, niter, best_ofv, validation_value);
            print_information(best_ofv, ctime, niter, validation_value);
        }
    }

    while (!stop) {
        // sort population by fitness
        std::sort(population.begin(), population.end(), 
            [](const training_individual& a, const training_individual& b) {
                return a.ofv > b.ofv;
            });

        std::vector<training_individual> new_population(population_size);

        // elites
        for (int i = 0; i < n_elites; ++i)
            new_population[i] = population[i];

        // mutants
        for (int i = 0; i < n_mutants && !stop; ++i) {
            auto& ind = new_population[n_elites + i];
            ind.weights.resize(n_weights);
            for (double& w : ind.weights)
                w = weight_dist(generator);

            apply_decoder(ind);

            ctime = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            if (ctime > training_time_limit) stop = true;

            if (ind.ofv > best_ofv) {
                best_ofv = ind.ofv;
                best_weights = ind.weights;
                write_weights_to_file(best_weights, ctime);
                double val = calculate_validation_value(best_weights);
                write_training_and_validation_values(training_file, validation_file, ctime, niter, best_ofv, val);
                print_information(best_ofv, ctime, niter, val);
            }
        }

        // offspring
        for (int i = 0; i < n_offspring && !stop; ++i) {
            auto& child = new_population[n_elites + n_mutants + i];
            child.weights.resize(n_weights);

            if (ga_config == 1) { // RKGA
                std::vector<int> idx(population_size);
                std::iota(idx.begin(), idx.end(), 0);
                std::shuffle(idx.begin(), idx.end(), generator);
                int p1 = idx[0], p2 = idx[1];

                for (int j = 0; j < n_weights; ++j)
                    child.weights[j] = (standard_distribution_01(generator) <= 0.5) ? population[p1].weights[j] : population[p2].weights[j];

            } else if (ga_config == 2) { // BRKGA
                int p1 = produce_random_integer(n_elites, standard_distribution_01(generator));
                int p2 = n_elites + produce_random_integer(population_size - n_elites, standard_distribution_01(generator));

                for (int j = 0; j < n_weights; ++j)
                    child.weights[j] = (standard_distribution_01(generator) <= elite_inheritance_probability) ? population[p1].weights[j] : population[p2].weights[j];

            } else if (ga_config == 3) { // Lexicase
                std::vector<training_individual> parents;
                for (int k = 0; k < 2; ++k) {
                    auto shuffled_instances = training_instances;
                    std::shuffle(shuffled_instances.begin(), shuffled_instances.end(), generator);
                    std::vector<training_individual> candidates;
                    for (auto& inst : shuffled_instances) {
                        double best_val = 0;
                        for (const auto& ind : population) {
                            store_weights(ind.weights);
                            double val = BS(training_bs_time_limit, training_beam_width, &inst, *this, true);
                            if (val >= best_val) {
                                if (val > best_val) {
                                    best_val = val;
                                    candidates.clear();
                                }
                                candidates.push_back(ind);
                            }
                        }
                    }
                    int sel = produce_random_integer(candidates.size(), standard_distribution_01(generator));
                    parents.push_back(candidates[sel]);
                }

                for (int j = 0; j < n_weights; ++j)
                    child.weights[j] = (standard_distribution_01(generator) <= 0.5) ? parents[0].weights[j] : parents[1].weights[j];
            }

            apply_decoder(child);

            ctime = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            if (ctime > training_time_limit) stop = true;

            if (child.ofv > best_ofv) {
                best_ofv = child.ofv;
                best_weights = child.weights;
                write_weights_to_file(best_weights, ctime);
                double val = calculate_validation_value(best_weights);
                write_training_and_validation_values(training_file, validation_file, ctime, niter, best_ofv, val);
                print_information(best_ofv, ctime, niter, val);
            }
        }

        population = std::move(new_population);
        ctime = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (ctime > training_time_limit) stop = true;
        ++niter;
    }

    std::cout << "------------ END OF TRAINING ------------" << std::endl;
    return best_weights;
}

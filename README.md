# RLCS-BS
Evolutionary offline learning framework applied to a Beam Search

To compile the program, use the provided Makefile.
Note: You must specify the path to the EIGEN library within the Makefile.

Solving an Instance
To solve an instance, use the following command-line parameters:

-i <path>: Path to the instance file to be solved.

-o <path>: Path to the output file where results will be stored.

-beam_width <int>: Beam width used during the search.

-time_limit <int>: Time limit for solving (in seconds).

-hidden_layers <int>: Number of hidden layers in the neural network.

-units <list>: Number of hidden units per hidden layer. Provide one integer per hidden layer.

-activation_function <int>: Activation function to use:

1 → tanh

2 → ReLU

3 → sigmoid

-feature_configuration <int>: Determines which features are used:

1 → Max, min, std and average of p^{L,v} and l^v, plus the length of the partial solution associated to node v are used (9 features)

2 → Same as 1, with alphabet size added.

3 → Same as 2, with number of input and restricted strings added.

4 → Same as 3, with string lengths added (assumes all strings are of equal length).

Important: A file named weights.txt containing trained network weights is required to solve an instance. This file must be in the same format as the one output by the training process.

Training the Neural Network
To train the network, the following parameters are available:

-hidden_layers <int>: Number of hidden layers in the neural network.

-units <list>: Number of hidden units per layer. Provide one integer per layer.

-weight_limit <int>: Maximum absolute value allowed for the network weights.

-training_beam_width <int>: Beam width used during training.

-training_time_limit <int>: Time limit for training (in seconds).

-activation_function <int>: Activation function to use (same options as for solving).

-feature_configuration <int>: Feature configuration (same as above).

-ga_configuration <int>: Genetic algorithm configuration:

1 → Standard RKGA

2 → BRKGA (requires specifying elite inheritance probability using -rho)

3 → RKGA with lexicase selection for elite population

Training and Validation Files
Training instances should be listed in a file named training_files.txt, one per line.

Validation instances should be listed in validation_files.txt, one per line.

The base path to these instance files should be specified in instances_path.txt.

Examples
Solving an Instance
./main -i ../instances/Rahman/converted/data_StrEC-converted/g15.txt \
       -weights weights.txt \
       -hidden_layers 2 \
       -units 10 5 \
       -o out-g15.txt \
       -beam_width 100 \
       -time_limit 10
Training
An example training setup can be found in the Training-Example directory.


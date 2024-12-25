#ifndef LAYER_HPP
#define LAYER_HPP
#include <random>

struct LAYER
{
    std::vector<std::vector<float>> weights;
    std::vector<float> biases;
    std::vector<float> weighted_sums;
    std::vector<float> outputs;
    std::vector<float> deltas;
};

/*
 * initialize layer weights and biases
 */
LAYER initialize_layer(int inputs, int neurons);

#endif
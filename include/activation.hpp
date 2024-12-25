#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP
#include <cmath>
#include <vector>
#include "../include/layer.hpp"

/*
 * ReLU activation for the hidden layer
 */
float relu(float x);

/*
 * softmax activation for the output layers
 * returns a vector of probabilities which sums to 1
 */
std::vector<float> softmax(LAYER *layer, int num_classes);

#endif
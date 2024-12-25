#include "../include/activation.hpp"
#include <algorithm>

float relu(float x)
{
    return fmax(0, x);
}

std::vector<float> softmax(LAYER *layer, int num_classes)
{
    // initialize output vector
    std::vector<float> output(num_classes, 0.0f);

    // find the maximum value in the outputs for numerical stability
    float max_value = *std::max_element(layer->outputs.begin(), layer->outputs.end());

    // compute exponentiated values and accumulate their sum
    float sum_exp = 0.0;
    for (int i = 0; i < num_classes; ++i)
    {
        // subtract max for stability
        float exp_value = std::exp(layer->outputs[i] - max_value);
        // store the exponentiated value in the output
        output[i] = exp_value;
        // accumulate the sum of exponentiated values
        sum_exp += exp_value;
    }

    // normalize the exponentiated values (to get probabilities)
    for (size_t i = 0; i < output.size(); ++i)
    {
        // normalize each value by the sum of exponentiated values
        output[i] /= sum_exp;
    }

    return output;
}
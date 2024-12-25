#include "../include/layer.hpp"

LAYER initialize_layer(int inputs, int neurons)
{
    LAYER layer;

    // use random device and normal distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / inputs));

    // initialize weights with He initialization
    layer.weights = std::vector<std::vector<float>>(neurons, std::vector<float>(inputs));
    for (int i = 0; i < neurons; ++i)
    {
        for (int j = 0; j < inputs; ++j)
        {
            layer.weights[i][j] = dist(gen);
        }
    }

    // initialize vectors with zeros
    layer.biases = std::vector<float>(neurons, 0.0f);
    layer.weighted_sums = std::vector<float>(neurons, 0.0f);
    layer.outputs = std::vector<float>(neurons, 0.0f);
    layer.deltas = std::vector<float>(neurons, 0.0f);

    return layer;
}
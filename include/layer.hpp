#ifndef LAYER_HPP
#define LAYER_HPP
#include <random>

struct LAYER
{
    std::vector<std::vector<float>> weights; // 2D vector
    std::vector<float> biases;
    std::vector<float> weighted_sums;
    std::vector<float> outputs;
    std::vector<float> deltas;

    /*
     * initialize layers weights and biases
     * with He initialization
     * where the values cluster around 0, with a standard deviation of sqrt(2 / inputs)
     */
    void initialize_layer(int inputs, int neurons)
    {
        // use random device and normal distribution
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / inputs));

        // define the size of the 2D vector
        this->weights = std::vector<std::vector<float>>(neurons, std::vector<float>(inputs));

        // initialize weights
        for (int i = 0; i < neurons; ++i)
        {
            for (int j = 0; j < inputs; ++j)
            {
                this->weights[i][j] = dist(gen);
            }
        }

        // initialize other vectors with zeros
        this->biases = std::vector<float>(neurons, 0.0f);
        this->weighted_sums = std::vector<float>(neurons, 0.0f);
        this->outputs = std::vector<float>(neurons, 0.0f);
        this->deltas = std::vector<float>(neurons, 0.0f);
    }
};

#endif
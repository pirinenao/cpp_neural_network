#include "../include/evaluation.hpp"

float sparse_cross_entropy_loss(std::vector<float> &softmax_output, int true_label)
{
    // get the predicted probability for the true class
    float predicted_probability = softmax_output[true_label];

    // add a small epsilon to prevent log(0)
    const float epsilon = 1e-10;
    predicted_probability = std::max(predicted_probability, epsilon); // Ensure it's not zero

    // compute the negative log of the predicted probability
    return -std::log(predicted_probability);
}

int max_value_index(std::vector<float> &vector)
{
    // Initialize the index and maximum value
    int max_index = -1;
    float max_value = 0.0f; // Start with the smallest possible integer

    // Loop through the vector to find the largest value
    for (size_t i = 0; i < vector.size(); ++i)
    {
        if (vector[i] > max_value)
        {
            max_value = vector[i];
            max_index = i;
        }
    }

    return max_index;
}
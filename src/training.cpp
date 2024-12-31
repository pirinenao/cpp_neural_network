#include "../include/training.hpp"
#include <thread>
#include <functional> // for std::ref

/*
 * computes the weighted sums for the neurons in the layer
 */
void forward_feed(LAYER *layer,
                  const std::vector<std::vector<float>> &images,
                  int sample_index, int neurons)
{
    // reset weighted sums for this forward pass
    std::fill(layer->weighted_sums.begin(), layer->weighted_sums.end(), 0.0f);

    for (int i = 0; i < neurons; i++)
    {
        // calculate weighted sum
        for (size_t j = 0; j < layer->weights[i].size(); j++)
        {
            layer->weighted_sums[i] += (layer->weights[i][j] * images[sample_index][j]);
        }

        // add bias and apply activation function (ReLU)
        layer->outputs[i] = relu(layer->weighted_sums[i] + layer->biases[i]);
    }
}

/*
 * computes the weighted sums for the neurons in the output layer
 */
void feed_output(LAYER *layer, LAYER *input_layer, int neurons)
{
    // initialize weighted sums vector
    std::fill(layer->weighted_sums.begin(), layer->weighted_sums.end(), 0.0f);

    for (int i = 0; i < neurons; i++)
    {
        // calculate weighted sum
        for (size_t j = 0; j < layer->weights[i].size(); j++)
        {
            layer->weighted_sums[i] += layer->weights[i][j] * input_layer->outputs[j];
        }

        // add bias
        // no activation function needed here for output layer, softmax will be applied outside
        layer->outputs[i] = layer->weighted_sums[i] + layer->biases[i];
    }
}

/*
 * parallel version of forward_feed
 */
void forward_feed_parallel(LAYER *layer,
                           const std::vector<std::vector<float>> &images,
                           int sample_index, int neurons)
{
    // reset weighted sums for this forward pass
    std::fill(layer->weighted_sums.begin(), layer->weighted_sums.end(), 0.0f);

    // define a lambda function to process a range of neurons
    auto process_neurons = [&](int start, int end)
    {
        for (int i = start; i < end; i++)
        {
            // calculate weighted sum
            for (size_t j = 0; j < layer->weights[i].size(); j++)
            {
                layer->weighted_sums[i] += (layer->weights[i][j] * images[sample_index][j]);
            }

            // add bias and apply activation function (ReLU)
            layer->outputs[i] = relu(layer->weighted_sums[i] + layer->biases[i]);
        }
    };

    // determine the number of threads to use
    const int num_threads = std::thread::hardware_concurrency();
    // calculate the chunk size for each thread
    // adding num_threads - 1 to round up
    const int chunk_size = (neurons + num_threads - 1) / num_threads;

    // create and launch threads
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; i++)
    {
        int start = i * chunk_size;
        int end = std::min(start + chunk_size, neurons);
        if (start < neurons)
        {
            threads.emplace_back(process_neurons, start, end);
        }
    }

    // join threads
    for (size_t i = 0; i < threads.size(); i++)
    {
        threads[i].join();
    }
}

/*
 * backpropagate the output layer
 * update the weights and biases based on the error
 */
void backpropagate_output(LAYER &layer, LAYER &input_layer, int expected_class, float learning_rate)
{
    std::vector<float> output_deltas(layer.outputs.size());

    // calculate deltas (the gradient propagated back)
    for (size_t i = 0; i < layer.outputs.size(); i++)
    {
        output_deltas[i] = layer.outputs[i] - (i == (size_t)expected_class ? 1.0f : 0.0f);
    }

    // update weights and biases for the output layer
    for (size_t i = 0; i < layer.outputs.size(); i++)
    {
        for (size_t j = 0; j < input_layer.outputs.size(); j++)
        {
            // update the weight based on gradient descent
            layer.weights[i][j] -= learning_rate * output_deltas[i] * input_layer.outputs[j];
        }

        // update the bias for the output neuron
        layer.biases[i] -= learning_rate * output_deltas[i];
        layer.deltas[i] = output_deltas[i];
    }
}

/*
 * backpropagate the hidden layer
 * update the weights and biases based on the error
 */
void backpropagate_hidden(LAYER &layer, LAYER &next_layer,
                          const mnist::MNIST_dataset<std::vector, std::vector<float>, int> &dataset, int sample_index, float learning_rate)
{
    std::vector<float> layer_errors(layer.outputs.size());
    std::vector<float> layer_deltas(layer.outputs.size());

    // initialize errors vector to 0
    std::fill(layer_errors.begin(), layer_errors.end(), 0.0f);

    // compute error for the hidden layer neurons
    for (size_t i = 0; i < layer.outputs.size(); i++)
    {
        // sum the errors weighted by the next layer's weights
        for (size_t j = 0; j < next_layer.deltas.size(); j++)
        {
            layer_errors[i] += next_layer.deltas[j] * next_layer.weights[j][i];
        }

        // calculate the delta for the layer
        layer_deltas[i] = layer_errors[i] * (layer.outputs[i] > 0 ? 1.0f : 0.0f);
    }

    // update weights and biases for the layer
    for (size_t i = 0; i < layer.weights.size(); i++)
    {
        for (size_t j = 0; j < layer.weights[i].size(); j++)
        {
            // update weight based on gradient descent
            layer.weights[i][j] -= learning_rate * layer_deltas[i] * dataset.training_images[sample_index][j];
        }

        // update biases (one bias per neuron in the hidden layer)
        layer.biases[i] -= learning_rate * layer_deltas[i];
    }
}
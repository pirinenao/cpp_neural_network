#ifndef TRAINING_HPP
#define TRAINING_HPP
#include "../mnist/mnist_reader.hpp"
#include "../activation.hpp"

/*
 * computes the weighted sums for the neurons in the layer
 */
void forward_feed(LAYER *layer,
                  const std::vector<std::vector<float>> &images,
                  int sample_index, int neurons);

/*
 * computes the weighted sums for the neurons in the output layer
 */
void feed_output(LAYER *layer, LAYER *input_layer, int neurons);

/*
 * uses parallel computing to compute the weighted sums for the neurons in the layer
 */
void forward_feed_parallel(LAYER *layer,
                           const std::vector<std::vector<float>> &images,
                           int sample_index, int neurons);

/*
 * backpropagate the output layer
 * update the weights and biases based on the error
 */
void backpropagate_output(LAYER &layer, LAYER &input_layer, int expected_class, float learning_rate);

/*
 * backpropagate the hidden layer
 * update the weights and biases based on the error
 */
void backpropagate_hidden(LAYER &layer, LAYER &next_layer,
                          const mnist::MNIST_dataset<std::vector, std::vector<float>, int> &dataset, int sample_index, float learning_rate);

#endif
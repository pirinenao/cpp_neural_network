#include <iostream>
#include <chrono>
#include "../include/mnist/mnist_reader.hpp"
#include "../include/mnist/mnist_utils.hpp"
#include "../include/evaluation.hpp"
#include "../include/model.hpp"

using namespace std;

#define NUM_INPUTS 784
#define NUM_WEIGHTS 784
#define NUM_NEURONS 128
#define NUM_OUTPUT_NEURONS 10
#define NUM_EPOCHS 1
#define LEARNING_RATE 0.001f

int main(void)
{
    // load the MNIST dataset
    mnist::MNIST_dataset<vector, vector<float>, int> dataset =
        mnist::read_dataset<vector, vector, float, int>(MNIST_DATA_LOCATION);

    // normalize the pixel values from 0-255 to 0-1
    mnist::normalize_pixels(dataset);

    // initialize layers and loss
    LAYER layer = initialize_layer(NUM_INPUTS, NUM_NEURONS);
    LAYER output_layer = initialize_layer(NUM_NEURONS, NUM_OUTPUT_NEURONS);
    EVALUATION eval;

    // train the model
    model_train(dataset, layer, output_layer, eval, NUM_EPOCHS, NUM_NEURONS, NUM_OUTPUT_NEURONS, LEARNING_RATE);
    model_evaluate(dataset, layer, output_layer, eval, NUM_NEURONS, NUM_OUTPUT_NEURONS);

    return 0;
}
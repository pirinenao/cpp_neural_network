#include <iostream>
#include <chrono>
#include "../include/mnist/mnist_reader.hpp"
#include "../include/mnist/mnist_utils.hpp"
#include "../include/utils.hpp"
#include "../include/progress_bar.hpp"

using namespace std;

#define NUM_INPUTS 784
#define NUM_WEIGHTS 784
#define NUM_NEURONS 128
#define NUM_CLASSES 10
#define NUM_EPOCHS 5
#define LEARNING_RATE 0.001f

int main(void)
{
    // load the MNIST dataset
    mnist::MNIST_dataset<vector, vector<float>, float> dataset =
        mnist::read_dataset<vector, vector, float, float>(MNIST_DATA_LOCATION);

    // normalize the pixel values from 255 to 0-1
    mnist::normalize_pixels(dataset);

    // initialize layers and loss
    LAYER layer = initialize_layer(NUM_INPUTS, NUM_NEURONS);
    LAYER output_layer = initialize_layer(NUM_NEURONS, NUM_CLASSES);
    LOSS loss = initialize_loss();

    // train the network
    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++)
    {
        // start timer
        auto start_time = std::chrono::high_resolution_clock::now();

        // iterate over the training set
        for (size_t sample_index = 0; sample_index < dataset.training_images.size(); sample_index++)
        {
            forward_feed(&layer, dataset, sample_index, NUM_NEURONS);
            feed_output(&output_layer, &layer, NUM_CLASSES);

            // perform softmax
            output_layer.outputs = softmax(&output_layer, NUM_CLASSES);

            // calculate loss
            loss.current_loss = sparse_cross_entropy_loss(output_layer.outputs, dataset.training_labels[sample_index]);
            loss.total_loss += loss.current_loss;
            loss.average_loss = (loss.average_loss * sample_index + loss.current_loss) / (sample_index + 1);

            // backpropagate the output layer
            backpropagate_output(output_layer, layer, (int)dataset.training_labels[sample_index], LEARNING_RATE);
            backpropagate_hidden(layer, output_layer, dataset, sample_index, LEARNING_RATE);

            // display progress (remove for faster training)
            if (sample_index % 100 == 0)
                progress_bar(sample_index, dataset.training_images.size(), epoch);
        }

        // end timer and display results
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end_time - start_time;

        cout << endl
             << "avg.loss: " << loss.average_loss
             << " learning rate: " << LEARNING_RATE
             << " time: " << elapsed.count() << " ms\n"
             << endl;

        loss = initialize_loss();
    }
}
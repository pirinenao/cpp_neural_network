#include "../include/mnist/mnist_reader.hpp"
#include "../include/mnist/mnist_utils.hpp"
#include "../include/layer.hpp"
#include "../include/evaluation.hpp"
#include "../include/model.hpp"
#include <unistd.h>

#define NUM_INPUTS 784
#define NUM_WEIGHTS 784
#define NUM_NEURONS 128
#define NUM_OUTPUT_NEURONS 10
#define NUM_EPOCHS 10
#define LEARNING_RATE 0.001f
#define PARALLEL_OFF 0
#define PARALLEL_ON 1

/*
 * print help message
 */
void print_help()
{
    std::cout << "Usage: ./main [options]\n\n"
              << "Options:\n"
              << "  -l <learning rate>  Specify the learning rate (positive float).\n"
              << "  -e <epochs>         Specify the number of epochs (positive integer).\n"
              << "  -p                  Enable parallel computing.\n"
              << "  -h                  Display this help message.\n"
              << std::endl;
}

int main(int argc, char **argv)
{
    int opt;
    int epochs = NUM_EPOCHS;
    int parallel = PARALLEL_OFF;
    float learning_rate = LEARNING_RATE;

    // handle CLI arguments
    while ((opt = getopt(argc, argv, "e:l:ph")) != -1)
    {
        switch (opt)
        {
        case 'e':
            epochs = std::atoi(optarg);
            if (epochs <= 0)
            {
                std::cout << "Error: Number of epochs must be a positive integer\n";
                return 1;
            }
            break;
        case 'l':
            learning_rate = std::atof(optarg);
            if (learning_rate <= 0)
            {
                std::cout << "Error: Learning rate must be a positive float\n";
                return 1;
            }
            break;
        case 'p':
            parallel = PARALLEL_ON;
            break;
        case 'h':
            print_help();
            return 0;
        default:
            std::cout << "Usage: ./main [options]\n";
            return 1;
        }
    }

    // load the MNIST dataset
    mnist::MNIST_dataset<std::vector, std::vector<float>, int> dataset =
        mnist::read_dataset<std::vector, std::vector, float, int>(MNIST_DATA_LOCATION);

    // normalize the pixel values from 0-255 to 0-1
    mnist::normalize_pixels(dataset);

    // initialize layers and evaluation struct
    LAYER layer;
    LAYER output_layer;
    layer.initialize_layer(NUM_INPUTS, NUM_NEURONS);
    output_layer.initialize_layer(NUM_NEURONS, NUM_OUTPUT_NEURONS);
    EVALUATION eval;

    // train the model
    model_train(dataset, layer, output_layer, eval, epochs, NUM_NEURONS, NUM_OUTPUT_NEURONS, learning_rate, parallel);
    model_evaluate(dataset, layer, output_layer, eval, NUM_NEURONS, NUM_OUTPUT_NEURONS, parallel);

    return 0;
}
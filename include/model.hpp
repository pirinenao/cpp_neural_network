#ifndef MODEL_HPP
#define MODEL_HPP
#include "../evaluation.hpp"
#include "../training.hpp"
#include "../progress_bar.hpp"

/**
 * trains the model using the training dataset
 */
void model_train(mnist::MNIST_dataset<std::vector, std::vector<float>, int> dataset, LAYER &layer,
                 LAYER &output_layer, EVALUATION eval, int num_epochs, int num_neurons, int num_classes, float learning_rate);

/*
 * evaluates model by using the validation dataset
 */
void model_evaluate(mnist::MNIST_dataset<std::vector, std::vector<float>, int> dataset, LAYER &layer,
                    LAYER &output_layer, EVALUATION eval, int num_neurons, int num_classes);

#endif
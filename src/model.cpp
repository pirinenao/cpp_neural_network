#include "../include/model.hpp"

/**
 * trains the model using the training dataset
 */
void model_train(mnist::MNIST_dataset<std::vector, std::vector<float>, int> dataset, LAYER &layer,
                 LAYER &output_layer, EVALUATION eval, int num_epochs, int num_neurons, int num_classes, float learning_rate)
{
    std::cout << "----------------------------------------"
              << std::endl;
    std::cout << "Training model on the training dataset\n";
    std::cout << "----------------------------------------"
              << std::endl;
    std::cout << "Number of samples: " << dataset.training_images.size() << std::endl;
    std::cout << "Number of epochs: " << num_epochs << std::endl;
    std::cout << "Learning rate: " << learning_rate << std::endl
              << std::endl;

    for (int epoch = 1; epoch <= num_epochs; epoch++)
    {
        eval.start_timer();
        // iterate over the training set
        for (size_t sample_index = 0; sample_index < dataset.training_images.size(); sample_index++)
        {
            forward_feed(&layer, dataset.training_images, sample_index, num_neurons);
            feed_output(&output_layer, &layer, num_classes);

            // perform softmax
            output_layer.outputs = softmax(&output_layer, num_classes);

            // calculate loss
            float loss = sparse_cross_entropy_loss(output_layer.outputs, dataset.training_labels[sample_index]);

            eval.set_loss(loss, sample_index);

            // backpropagate the output layer
            backpropagate_output(output_layer, layer, (int)dataset.training_labels[sample_index], learning_rate);
            backpropagate_hidden(layer, output_layer, dataset, sample_index, learning_rate);

            // display progress (remove for faster training)
            if (sample_index % 1000 == 0)
            {
                progress_bar(sample_index, dataset.training_images.size(), epoch);
            }
        }

        eval.end_timer();
        eval.print_training_metrics();
        eval.initialize_loss();
    }
}

/*
 * evaluates model by using the validation dataset
 */
void model_evaluate(mnist::MNIST_dataset<std::vector, std::vector<float>, int> dataset, LAYER &layer,
                    LAYER &output_layer, EVALUATION eval, int num_neurons, int num_classes)
{
    std::cout << "----------------------------------------"
              << std::endl;
    std::cout << "Evaluating model on the validation dataset\n";
    std::cout << "----------------------------------------"
              << std::endl;
    std::cout << "Number of samples: " << dataset.test_images.size() << std::endl
              << std::endl;

    eval.initialize_loss();
    std::vector<int> predictions;

    for (size_t sample_index = 0; sample_index < dataset.test_images.size(); sample_index++)
    {
        forward_feed_parallel(&layer, dataset.test_images, sample_index, num_neurons);
        feed_output(&output_layer, &layer, num_classes);

        // perform softmax
        output_layer.outputs = softmax(&output_layer, num_classes);

        int prediction = max_value_index(output_layer.outputs);
        predictions.push_back(prediction);

        // calculate loss
        float loss = sparse_cross_entropy_loss(output_layer.outputs, dataset.test_labels[sample_index]);
        eval.set_loss(loss, sample_index);

        // display progress (remove for faster training)
        if (sample_index % 100 == 0)
        {
            progress_bar(sample_index, dataset.test_images.size(), NO_EPOCHS);
        }
    }

    eval.set_labels(predictions, dataset.test_labels);
    eval.print_metrics();
    eval.display_confusion_matrix(num_classes);
    eval.display_precision(num_classes);
}
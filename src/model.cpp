#include "../include/model.hpp"

/**
 * trains the model using the training dataset
 */
void model_train(mnist::MNIST_dataset<std::vector, std::vector<float>, int> dataset, LAYER &layer,
                 LAYER &output_layer, EVALUATION eval, int num_epochs, int num_neurons, int num_classes, float learning_rate)
{
    std::cout << "Training model on training dataset\n"
              << std::endl;
    std::cout << "Number of samples: " << dataset.training_images.size() << std::endl;
    std::cout << "Number of epochs: " << num_epochs << std::endl;
    std::cout << "Learning rate: " << learning_rate << std::endl
              << std::endl;

    for (int epoch = 1; epoch <= num_epochs; epoch++)
    {
        // start timer
        auto start_time = std::chrono::high_resolution_clock::now();

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

        // end timer and display results
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end_time - start_time;
        std::cout << std::endl
                  << "avg.loss: " << eval.average_loss
                  << " time: " << elapsed.count() << " ms\n"
                  << std::endl;

        eval.initialize_loss();
    }
}

/*
 * evaluates model by using the validation dataset
 */
void model_evaluate(mnist::MNIST_dataset<std::vector, std::vector<float>, int> dataset, LAYER &layer,
                    LAYER &output_layer, EVALUATION eval, int num_neurons, int num_output_neurons)
{
    eval.initialize_loss();
    std::vector<int> predictions;

    std::cout << "Evaluating model on validation dataset\n"
              << std::endl
              << "Number of samples: " << dataset.test_images.size() << std::endl
              << std::endl;

    for (size_t sample_index = 0; sample_index < dataset.test_images.size(); sample_index++)
    {
        forward_feed(&layer, dataset.test_images, sample_index, num_neurons);
        feed_output(&output_layer, &layer, num_output_neurons);

        // perform softmax
        output_layer.outputs = softmax(&output_layer, num_output_neurons);

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
    std::cout << std::endl
              << "avg.loss: " << eval.average_loss << std::endl
              << "accuracy: " << eval.accuracy() << std::endl;
}
#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>

#include "activation_functions.hpp"
#include "dataset.hpp"
#include "fully_connected.hpp"
#include "random.hpp"
#include "sequence.hpp"

int
main()
{
  constexpr int epochs = 20;
  constexpr std::size_t batch_size = 200;
  constexpr float initial_learning_rate = 1e-4f;
  constexpr float gamma = 0.95;
  constexpr float rms_prop_smoothing_factor = 1e-8f;
  constexpr float rms_prop_history_influence = 0.9f;
  constexpr float validation_dataset_fraction = 0.1f;
  constexpr std::size_t seed = 1231331231231231;

  auto start_time = std::chrono::system_clock::now();

  auto random = nnets::Random{};
  random.seed(seed);

  // Read train dataset
  auto train_dataset =
    nnets::read_dataset("data/fashion_mnist_train_vectors.csv",
                        "data/fashion_mnist_train_labels.csv");
  const auto full_train_dataset = train_dataset;
  auto input_vector_size = train_dataset.at(0).first.size();
  auto num_categories = nnets::num_categories(train_dataset);
  std::cout << "train_dataset_size=" << train_dataset.size() << "\n";
  std::cout << "input_vector_size=" << input_vector_size << "\n";
  std::cout << "num_categories=" << num_categories << "\n";

  // Reserve part of train data for validation
  std::ranges::shuffle(train_dataset, random.rng());
  auto validation_data_start =
    train_dataset.begin() +
    static_cast<std::size_t>((1.0f - validation_dataset_fraction) *
                             train_dataset.size());
  auto validation_dataset =
    nnets::Dataset{ validation_data_start, train_dataset.end() };
  train_dataset.erase(validation_data_start, train_dataset.end());

  // Network topology
  auto net = nnets::Sequence{ {
    std::make_shared<nnets::FullyConnected<nnets::RelU>>(input_vector_size,
                                                         300),
    std::make_shared<nnets::FullyConnected<nnets::RelU>>(300, 200),
    std::make_shared<nnets::FullyConnected<nnets::RelU>>(200, 100),
    std::make_shared<nnets::FullyConnected<nnets::RelU>>(100, num_categories),
  } };
  net.init_weights(random);

  // Helper arrays
  auto expected_vector = std::vector<float>(num_categories);
  auto error_grad = std::vector<float>(num_categories);

  // Will be lowered progressively
  float learning_rate = initial_learning_rate;

  // Pass through the dataset in epochs
  for (int epoch = 0; epoch < epochs; ++epoch) {
    std::ranges::shuffle(train_dataset, random.rng());

    int batch = 0;
    float error = 0.0f;

    // Split dataset into mini-batches
    for (std::size_t batch_start = 0; batch_start < train_dataset.size();
         batch_start += batch_size, ++batch) {
      std::size_t current_batch_size =
        std::min(batch_size, train_dataset.size() - batch_start);

      net.zero_grad();

      float batch_error = 0.0f;

      // Iterate over batch
      for (std::size_t i = 0; i < current_batch_size; ++i) {
        // Forward feed
        const auto& [input, expected_label] = train_dataset[batch_start + i];
        net.forward(input);
        const auto output = net.output();

        // Compute error
        expected_vector.assign(num_categories, 0.0f);
        expected_vector[expected_label] = 1.0f;

        for (std::size_t k = 0; k < num_categories; ++k) {
          error_grad[k] = output[k] - expected_vector[k];
          batch_error += 0.5f * error_grad[k] * error_grad[k];
        }

        // Backpropagation
        net.backward(error_grad);
      }

      // Learning step
      net.step_grad_rms_prop(
        learning_rate, rms_prop_history_influence, rms_prop_smoothing_factor);

      std::cout << "epoch=" << epoch << " batch=" << batch
                << " batch_error=" << batch_error << std::endl;
    }

    // Lower the learning rate
    learning_rate *= gamma;

    // Evaluate classification success on validation data after epoch
    int success_count = 0;

    for (const auto& [input, expected] : validation_dataset) {
      net.forward(input);
      auto output = net.output();
      auto max_category =
        std::distance(output.begin(), std::ranges::max_element(output));

      if (max_category == expected) {
        ++success_count;
      }
    }

    float success_rate =
      static_cast<float>(success_count) / validation_dataset.size();

    std::cout << "epoch=" << epoch << " success_rate=" << success_rate
              << std::endl;
  }

  // Evaluate full train dataset
  int train_success_count = 0;
  auto train_predictions = std::vector<int>{};

  for (const auto& [input, expected] : full_train_dataset) {
    net.forward(input);
    auto output = net.output();
    auto max_category =
      std::distance(output.begin(), std::ranges::max_element(output));

    train_predictions.push_back(static_cast<int>(max_category));

    if (max_category == expected) {
      ++train_success_count;
    }
  }

  float train_success_rate =
    static_cast<float>(train_success_count) / full_train_dataset.size();
  std::cout << "final train dataset success rate " << train_success_rate
            << std::endl;
  nnets::write_predictions("trainPredictions", train_predictions);

  // Read and evaluate test dataset
  const auto test_dataset =
    nnets::read_dataset("data/fashion_mnist_test_vectors.csv",
                        "data/fashion_mnist_test_labels.csv");
  int test_success_count = 0;
  auto test_predictions = std::vector<int>{};

  for (const auto& [input, expected] : test_dataset) {
    net.forward(input);
    auto output = net.output();
    auto max_category =
      std::distance(output.begin(), std::ranges::max_element(output));

    test_predictions.push_back(static_cast<int>(max_category));

    if (max_category == expected) {
      ++test_success_count;
    }
  }

  float test_success_rate =
    static_cast<float>(test_success_count) / test_dataset.size();
  std::cout << "final test dataset success rate " << test_success_rate
            << std::endl;
  nnets::write_predictions("actualTestPredictions", test_predictions);

  auto end_time = std::chrono::system_clock::now();
  std::cout << "Total runtime: "
            << std::chrono::duration_cast<std::chrono::seconds>(end_time -
                                                                start_time)
                 .count()
            << " seconds" << std::endl;

  return 0;
}

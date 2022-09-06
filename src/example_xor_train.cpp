#include <array>
#include <iostream>
#include <random>
#include <utility>

#include "random.hpp"
#include "xor_net.hpp"

int
main()
{
  auto rdev = std::random_device{};
  auto random = nnets::Random{};
  random.seed(rdev());

  auto net = nnets::XorNet{};
  net.init_weights(random);

  auto train_data = std::array{
    std::pair{ std::array{ 0.0f, 0.0f }, std::array{ 0.0f } },
    std::pair{ std::array{ 0.0f, 1.0f }, std::array{ 1.0f } },
    std::pair{ std::array{ 1.0f, 0.0f }, std::array{ 1.0f } },
    std::pair{ std::array{ 1.0f, 1.0f }, std::array{ 0.0f } },
  };

  int epochs = 10'000;
  float learning_rate = 0.5f;

  for (int i = 0; i < epochs; ++i) {
    float error = 0.0f;

    net.zero_grad();

    for (auto [input, expected] : train_data) {
      net.forward(input);

      float error_grad = net.output()[0] - expected[0];
      error += 0.5f * error_grad * error_grad;

      net.backward(std::array{ error_grad });
    }

    net.step_grad(learning_rate);

    std::cout << "epoch=" << i << "; error=" << error << "\n";
  }

  for (auto [input, expected] : train_data) {
    input[0] += 0.1f;
    input[1] -= 0.1f;
    net.forward(input);
    std::cout << "x0=" << input[0] << " x1=" << input[1]
              << " y=" << net.output()[0] << "\n";
  }
}

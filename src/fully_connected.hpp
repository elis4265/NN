#pragma once

#include <algorithm>
#include <cstddef>
#include <span>
#include <vector>

#include "module.hpp"

namespace nnets {

// Fully connected layer
// See IModule for method documentation
// See activation_functions.hpp for possible ActivationFn types
template<typename ActivationFn>
class FullyConnected : public IModule
{
public:
  FullyConnected(std::size_t input_size,
                 std::size_t output_size,
                 ActivationFn activation_fn = {})
    : input_size_{ input_size }
    , output_size_{ output_size }
    , activation_fn_{ activation_fn }
  {
    bias_.resize(output_size);
    weights_.resize(input_size * output_size);
    potential_.resize(output_size);
    output_.resize(output_size);
    output_derivative_.resize(output_size);
    input_grad_.resize(input_size);
    bias_grad_.resize(output_size);
    bias_grad_history_.resize(output_size);
    weight_grad_.resize(input_size * output_size);
    weight_grad_history_.resize(input_size * output_size);
  }

  void
  forward(std::span<const float> input) override
  {
    input_ = input;

    for (std::size_t j = 0; j < output_size_; ++j) {
      potential_[j] = bias_[j];
      for (std::size_t i = 0; i < input_size_; ++i) {
        potential_[j] += weights_[j * input_size_ + i] * input[i];
      }
    }

    for (std::size_t j = 0; j < output_size_; ++j) {
      output_[j] = activation_fn_(potential_[j]);
    }
  }

  void
  backward(std::span<const float> output_grad) override
  {
    for (std::size_t j = 0; j < output_size_; ++j) {
      output_derivative_[j] = activation_fn_.derivative(potential_[j]);
    }

    for (std::size_t j = 0; j < output_size_; ++j) {
      bias_grad_[j] += output_grad[j] * output_derivative_[j];

      for (std::size_t i = 0; i < input_size_; ++i) {
        weight_grad_[j * input_size_ + i] +=
          output_grad[j] * output_derivative_[j] * input_[i];
      }
    }

    for (std::size_t j = 0; j < input_size_; ++j) {
      input_grad_[j] = 0.0f;

      for (std::size_t r = 0; r < output_size_; ++r) {
        input_grad_[j] += output_grad[r] * output_derivative_[r] *
                          weights_[r * input_size_ + j];
      }
    }
  }

  void
  init_weights(Random& random) override
  {
    random.generate_normal(
      weights_, 0.0f, std::sqrt(2.0f / (input_size_ * output_size_)));
  }

  void
  zero_grad() override
  {
    bias_grad_.assign(output_size_, 0.0f);
    weight_grad_.assign(input_size_ * output_size_, 0.0f);
  }

  void
  step_grad(float learning_rate) override
  {
    for (std::size_t j = 0; j < output_size_; ++j) {
      bias_[j] -= learning_rate * bias_grad_[j];

      for (std::size_t i = 0; i < input_size_; ++i) {
        weights_[j * input_size_ + i] -=
          learning_rate * weight_grad_[j * input_size_ + i];
      }
    }
  }

  void
  step_grad_rms_prop(float learning_rate,
                     float history_influence,
                     float smoothing_term) override
  {
    for (std::size_t j = 0; j < output_size_; ++j) {
      bias_grad_history_[j] =
        history_influence * bias_grad_history_[j] +
        (1.0f - history_influence) * bias_grad_[j] * bias_grad_[j];
      bias_[j] -=
        (learning_rate / std::sqrt(bias_grad_history_[j] + smoothing_term)) *
        bias_grad_[j];

      for (std::size_t i = 0; i < input_size_; ++i) {
        weight_grad_history_[j * input_size_ + i] =
          history_influence * weight_grad_history_[j * input_size_ + i] +
          (1.0f - history_influence) * weight_grad_[j * input_size_ + i] *
            weight_grad_[j * input_size_ + i];
        weights_[j * input_size_ + i] -=
          (learning_rate / std::sqrt(weight_grad_history_[j * input_size_ + i] +
                                     smoothing_term)) *
          weight_grad_[j * input_size_ + i];
      }
    }
  }

  [[nodiscard]] std::span<const float>
  output() const override
  {
    return output_;
  }

  [[nodiscard]] std::span<const float>
  input_grad() const override
  {
    return input_grad_;
  }

  [[nodiscard]] std::span<float>
  weights()
  {
    return weights_;
  };

  std::span<float>
  bias()
  {
    return bias_;
  }

private:
  std::size_t input_size_;
  std::size_t output_size_;
  ActivationFn activation_fn_;
  std::vector<float> bias_;
  std::vector<float> weights_;
  std::vector<float> potential_;
  std::vector<float> output_;
  std::vector<float> output_derivative_;
  std::vector<float> input_grad_;
  std::vector<float> bias_grad_;
  std::vector<float> bias_grad_history_;
  std::vector<float> weight_grad_;
  std::vector<float> weight_grad_history_;
  std::span<const float> input_;
};

}

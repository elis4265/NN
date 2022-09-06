#pragma once

#include <span>

#include "random.hpp"

namespace nnets {

// Interface for neural network components
class IModule
{
public:
  virtual ~IModule() = default;

  // Activate the network (access activation results with output())
  virtual void
  forward(std::span<const float> input) = 0;

  // Backpropagation (access input gradient with backward())
  virtual void
  backward(std::span<const float> activation_gradient) = 0;

  // Reset accumulated weight gradient values to 0
  virtual void
  zero_grad() = 0;

  // Add learning_rate * accumulated gradient from backward() calls
  // to weights (SGD learning step)
  virtual void
  step_grad(float learning_rate) = 0;

  // RMSProp learning step
  virtual void
  step_grad_rms_prop(float learning_rate,
                     float history_influence,
                     float smoothing_term) = 0;

  // Random weights initialization
  virtual void
  init_weights(Random& random) = 0;

  // Activation results from the last call to forward()
  [[nodiscard]] virtual std::span<const float>
  output() const = 0;

  // Gradient of error function for inputs from the last call to backward()
  [[nodiscard]] virtual std::span<const float>
  input_grad() const = 0;
};

}

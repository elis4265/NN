#pragma once

#include "activation_functions.hpp"
#include "fully_connected.hpp"
#include "sequence.hpp"

namespace nnets {

// Simple network for learning the XOR function
class XorNet : public IModule
{
public:
  XorNet()
  {
    auto hidden_layer = std::make_shared<FullyConnected<LogisticSigmoid>>(2, 2);
    auto output_layer = std::make_shared<FullyConnected<LogisticSigmoid>>(2, 1);

    sequence_ = Sequence{ {
      hidden_layer,
      output_layer,
    } };
  }

  // Instead of training, setup known good weights
  void
  set_correct_weights()
  {
    auto hidden_layer = std::make_shared<FullyConnected<UnitStep>>(2, 2);
    auto output_layer = std::make_shared<FullyConnected<UnitStep>>(2, 1);

    hidden_layer->weights()[0] = 2;
    hidden_layer->weights()[1] = 2;
    hidden_layer->weights()[2] = -2;
    hidden_layer->weights()[3] = -2;
    hidden_layer->bias()[0] = -1;
    hidden_layer->bias()[1] = 3;

    output_layer->weights()[0] = 1;
    output_layer->weights()[1] = 1;
    output_layer->bias()[0] = -2;

    sequence_ = Sequence{ {
      hidden_layer,
      output_layer,
    } };
  }

  void
  forward(std::span<const float> input) override
  {
    sequence_.forward(input);
  }

  void
  backward(std::span<const float> output_grad) override
  {
    sequence_.backward(output_grad);
  }

  void
  zero_grad() override
  {
    sequence_.zero_grad();
  }

  void
  init_weights(Random& random) override
  {
    sequence_.init_weights(random);
  }

  void
  step_grad(float coeff) override
  {
    sequence_.step_grad(coeff);
  }

  void
  step_grad_rms_prop(float learning_rate,
                     float history_influence,
                     float smoothing_term) override
  {
    sequence_.step_grad_rms_prop(
      learning_rate, history_influence, smoothing_term);
  }

  [[nodiscard]] std::span<const float>
  output() const override
  {
    return sequence_.output();
  }

  [[nodiscard]] std::span<const float>
  input_grad() const override
  {
    return sequence_.input_grad();
  }

private:
  Sequence sequence_;
};

}
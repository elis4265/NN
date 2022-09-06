#pragma once

#include <memory>
#include <ranges>
#include <span>

#include "module.hpp"

namespace nnets {

// Neural network layers arranged in a linear sequence
class Sequence : public IModule
{
public:
  Sequence() = default;

  explicit Sequence(std::vector<std::shared_ptr<IModule>> modules)
    : modules_{ std::move(modules) }
  {}

  void
  forward(std::span<const float> input) override
  {
    for (auto& module : modules_) {
      module->forward(input);
      input = module->output();
    }
  }

  void
  backward(std::span<const float> output_grad) override
  {
    for (auto& module : modules_ | std::views::reverse) {
      module->backward(output_grad);
      output_grad = module->input_grad();
    }
  }

  void
  zero_grad() override
  {
    for (auto& module : modules_) {
      module->zero_grad();
    }
  }

  void
  init_weights(Random& random) override
  {
    for (auto& module : modules_) {
      module->init_weights(random);
    }
  }

  void
  step_grad(float coeff) override
  {
    for (auto& module : modules_) {
      module->step_grad(coeff);
    }
  }

  void
  step_grad_rms_prop(float learning_rate,
                     float history_influence,
                     float smoothing_term) override
  {
    for (auto& module : modules_) {
      module->step_grad_rms_prop(
        learning_rate, history_influence, smoothing_term);
    }
  }

  [[nodiscard]] std::span<const float>
  output() const override
  {
    return modules_.back()->output();
  }

  [[nodiscard]] std::span<const float>
  input_grad() const override
  {
    return modules_.front()->input_grad();
  }

private:
  std::vector<std::shared_ptr<IModule>> modules_;
};

}

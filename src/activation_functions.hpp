#pragma once

#include <cmath>

namespace nnets {

struct RelU {
  float operator()(float x) const {
    return std::max(0.0f, x);
  }

  float derivative(float x) const {
    return x >= 0.0f ? 1.0f : 0.0f;
  }
};

struct UnitStep {
  float operator()(float x) const {
    return x >= 0.0f ? 1.0f : 0.0f;
  }

  float derivative(float x) const {
    return 0.0f;
  }
};

struct LogisticSigmoid {
  float lambda = 1.0f;

  float operator()(float x) const {
    return 1.0f / (1.0f + std::exp(lambda * -x));
  }

  float derivative(float x) const {
    float val = (*this)(x);
    return val * (1.0f - val);
  }
};


}
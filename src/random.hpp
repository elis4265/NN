#pragma once

#include <algorithm>
#include <functional>
#include <random>
#include <span>

namespace nnets {

// RNG functionality
class Random
{
public:
  // Seed the RNG
  void
  seed(std::uint64_t seed)
  {
    rng_.seed(seed);
  }

  // Sample an uniform distribution
  void
  generate_uniform(std::span<float> result, float min, float max)
  {
    auto dist = std::uniform_real_distribution<float>{ min, max };
    std::ranges::generate(result, [&]() { return dist(rng_); });
  }

  // Sample a normal distribution
  void
  generate_normal(std::span<float> result, float mean, float stdev)
  {
    auto dist = std::normal_distribution<float>{ mean, stdev };
    std::ranges::generate(result, [&]() { return dist(rng_); });
  }

  // Access the underlying RNG engine
  auto&
  rng()
  {
    return rng_;
  }

private:
  std::mt19937_64 rng_;
};

}

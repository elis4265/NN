#include <array>
#include <iostream>

#include "xor_net.hpp"

int
main()
{
  auto net = nnets::XorNet{};
  net.set_correct_weights();

  auto inputs = std::array{ 0.0f, 0.0f };
  net.forward(inputs);

  std::cout << net.output()[0] << std::endl;
}

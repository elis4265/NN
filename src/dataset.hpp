#pragma once

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace nnets {

// Dataset is a set of pairs of input vectors and expected output categories
using Dataset = std::vector<std::pair<std::vector<float>, int>>;

// "0,1,0,2" -> {0,1,0,2}
[[nodiscard]] inline std::vector<float>
parse_input_vector(const std::string& input)
{
  auto result = std::vector<float>{};
  std::size_t start = 0;

  do {
    std::size_t end = input.find_first_of(',', start);
    std::size_t n = std::string::npos;
    if (end != std::string::npos) {
      n = end - start;
    }

    auto val_chars = input.substr(start, n);
    if (val_chars.empty()) {
      break;
    }

    int value = std::atoi(val_chars.c_str());

    result.push_back(static_cast<float>(value));

    start = end;
    if (start != std::string::npos) {
      ++start;
    }
  } while (start != std::string::npos);

  return result;
}

// Read dataset from vectors file and expected categories file
[[nodiscard]] inline Dataset
read_dataset(const std::filesystem::path& inputs_path,
             const std::filesystem::path& outputs_path)
{
  auto dataset = Dataset{};
  auto inputs_file = std::ifstream{ inputs_path };
  auto outputs_file = std::ifstream{ outputs_path };
  auto input_line = std::string{};
  auto output_line = std::string{};
  while (std::getline(inputs_file, input_line) and
         std::getline(outputs_file, output_line)) {
    auto input_vector = parse_input_vector(input_line);

    if (input_vector.empty()) {
      break;
    }

    dataset.emplace_back(std::move(input_vector),
                         std::atoi(output_line.c_str()));
  }

  return dataset;
}

// Write predictions into a file
inline void
write_predictions(const std::filesystem::path& predictions_path,
                  std::span<const int> predictions)
{
  auto predictions_file = std::ofstream{ predictions_path };

  for (int value : predictions) {
    predictions_file << value << "\n";
  }
}

// Count the total number of categories in a dataset
inline int
num_categories(const Dataset& dataset)
{
  int max_cat = 0;
  for (const auto& [input, label] : dataset) {
    max_cat = std::max(max_cat, label);
  }
  return max_cat + 1;
}

}

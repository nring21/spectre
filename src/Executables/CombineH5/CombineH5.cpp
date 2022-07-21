// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <boost/program_options.hpp>
#include <cstddef>
#include <map>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TensorData.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/VolumeData.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/Printf.hpp"
#include "Utilities/FileSystem.hpp"

// Charm looks for this function but since we build without a main function or
// main module we just have it be empty
extern "C" void CkRegisterMainModule(void) {}

size_t data_size(std::variant<DataVector, std::vector<float>> data) {
  if (data.index() == 0) {
    return std::get<DataVector>(data).size();
  } else {
    return std::get<std::vector<float>>(data).size();
  }
}

TensorComponent group_tensor_component(
    std::string name, std::variant<DataVector, std::vector<float>> data,
    size_t index, size_t group_length) {
  if (data.index() == 0) {
    DataVector group(group_length);
    for (size_t i = 0; i < group_length; ++i) {
      group[i] = std::get<DataVector>(data)[i + index];
    }
    return {name, group};
  } else {
    std::vector<float> group;
    for (size_t i = 0; i < group_length; ++i) {
      group.push_back(std::get<std::vector<float>>(data)[i + index]);
    }
    return {name, group};
  }
}

void combine_h5(const std::string& file_prefix, const std::string& subfile_name,
                const std::string& output) {
  // Parses for and stores all input files to be looped over
  std::vector<std::string> file_names = file_system::glob(file_prefix + "*.h5");

  // Instantiates the output file and the .vol subfile to be filled with the
  // combined data later
  h5::H5File<h5::AccessType::ReadWrite> new_file(output + "0.h5", true);
  new_file.insert<h5::VolumeData>("/" + subfile_name + ".vol");
  new_file.close_current_object();

  // std::map to store the sorted volume data
  std::map<int, std::vector<
                    std::tuple<size_t, double, std::vector<ElementVolumeData>>>>
      file_map;

  // Opens the original files, sorts the volume data, and stores it in the
  // std::map
  for (size_t i = 0; i < file_names.size(); ++i) {
    h5::H5File<h5::AccessType::ReadWrite> initial_file(
        file_prefix + std::to_string(i) + ".h5", true);

    auto& volume_file = initial_file.get<h5::VolumeData>("/" + subfile_name);

    auto sorted_volume_data = volume_file.get_data_by_element(
        std::nullopt, std::nullopt, std::nullopt);

    file_map.insert(
        std::pair<int, std::vector<std::tuple<size_t, double,
                                              std::vector<ElementVolumeData>>>>(
            i, sorted_volume_data));
    initial_file.close_current_object();
  }

  // Append vectors of ElementVolumeData that were split between H5 files for a
  // given observation id and value
  for (size_t i = 1; i < file_names.size(); ++i) {
    for (size_t j = 0; j < file_map.at(0).size(); ++j) {
      std::get<2>(file_map.at(0)[j])
          .insert(std::get<2>(file_map.at(0)[j]).end(),
                  std::get<2>(file_map.at(i)[j]).begin(),
                  std::get<2>(file_map.at(i)[j]).end());
    }
  }

  // Write combined volume data for the output file
  auto& new_edited_file = new_file.get<h5::VolumeData>("/" + subfile_name);
  for (auto const& [observation_id, observation_value, element_data] :
       file_map.at(0)) {
    new_edited_file.write_volume_data(observation_id, observation_value,
                                      element_data);
  }

  new_file.close_current_object();
}

int main(int argc, char** argv) {
  boost::program_options::positional_options_description pos_desc;

  boost::program_options::options_description desc("Options");
  desc.add_options()("help,h,", "show this help message")(
      "file_prefix", boost::program_options::value<std::string>()->required(),
      "prefix of the files to be combined (omit number and file extension)")(
      "subfile_name", boost::program_options::value<std::string>()->required(),
      "subfile name shared for each volume file in each H5 file (omit file "
      "extension)")("output",
                    boost::program_options::value<std::string>()->required(),
                    "combined output filename (omit file extension)");

  boost::program_options::variables_map vars;

  boost::program_options::store(
      boost::program_options::command_line_parser(argc, argv)
          .positional(pos_desc)
          .options(desc)
          .run(),
      vars);

  if (vars.count("help") != 0u or vars.count("file_prefix") == 0u or
      vars.count("subfile_name") == 0u or vars.count("output") == 0u) {
    Parallel::printf("%s\n", desc);
    return 0;
  }

  combine_h5(vars["file_prefix"].as<std::string>(),
             vars["subfile_name"].as<std::string>(),
             vars["output"].as<std::string>());
}

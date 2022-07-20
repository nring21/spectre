// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <boost/program_options.hpp>
#include <cstddef>
#include <cstdint>  // do we need???
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TensorData.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/VolumeData.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/Printf.hpp"

#include <iostream>
#include <typeinfo>
#include <variant>

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

void combine_h5(const std::string& file_prefix,
                const std::string& subfile_prefix,
                const std::string& output_prefix) {
  // Make new_file
  h5::H5File<h5::AccessType::ReadWrite> new_file(output_prefix + ".h5", true);
  // Add .vol path
  auto& new_volume_file =
      new_file.insert<h5::VolumeData>("/" + subfile_prefix + ".vol");
  new_file.close_current_object();

  // Open original file
  h5::H5File<h5::AccessType::ReadWrite> initial_file(file_prefix + "0" + ".h5",
                                                     true);
  auto& volume_file = initial_file.get<h5::VolumeData>("/" + subfile_prefix);

  auto read_observation_ids = volume_file.list_observation_ids();
  for (size_t i = 0; i < read_observation_ids.size(); ++i) {
    auto read_observation_value =
        volume_file.get_observation_value(read_observation_ids[i]);
    auto read_tensor_components =
        volume_file.list_tensor_components(read_observation_ids[i]);
    std::vector<TensorComponent> tensor_component;
    for (size_t j = 0; j < read_tensor_components.size(); ++j) {
      tensor_component.push_back(volume_file.get_tensor_component(
          read_observation_ids[i], read_tensor_components[j]));
    }
    std::vector<std::vector<TensorComponent>> modified_tensor_component;
    std::cout << tensor_component[0] << "\n";
    // for (size_t j = 0; j < tensor_component.size(); ++j) {
    //   std::string component_name = tensor_component[j].name;
    //   auto data = tensor_component[j].data;
    //   auto sub_vector_length = data_size(data)/read_extents.size();
    //   std::vector<TensorComponent> sub_vector;
    //   for (size_t k = 0; k < read_extents.size(); ++k) {

    //   }
    // }
    for (size_t j = 0; j < read_extents.size(); ++j) {
      std::vector<TensorComponent> sub_vector;
      for (size_t k = 0; k < tensor_component.size(); ++k) {
        std::string component_name = tensor_component[k].name;
        auto data = tensor_component[k].data;
        auto sub_vector_length = data_size(data) / read_extents.size();
        for (size_t l = 0; l < sub_vector_length; ++l) {
          sub_vector.emplace_back(data.get(
              l))  // Need a proper helper function depending on type in variant
        }
      }
    }
    auto read_bases = volume_file.get_bases(read_observation_ids[i]);
    auto read_quadratures =
        volume_file.get_quadratures(read_observation_ids[i]);
    auto read_extents = volume_file.get_extents(read_observation_ids[i]);
    auto read_grid_names = volume_file.get_grid_names(read_observation_ids[i]);

    initial_file.close_current_object();

    std::vector<std::vector<Spectral::Basis>> vector_of_bases;
    // Make correct vector of Spectral::Basis::[name of basis]
    for (size_t j = 0; j < read_bases.size(); ++j) {
      std::vector<Spectral::Basis> temp_basis;
      for (size_t k = 0; k < read_bases[j].size(); ++k) {
        temp_basis.push_back(Spectral::to_basis(read_bases[j][k]));
      }
      vector_of_bases.push_back(temp_basis);
    }

    std::vector<std::vector<Spectral::Quadrature>> vector_of_quadrature;
    // Make correct vector of Spectral::Quadrature::[name of quadrature]
    for (size_t j = 0; j < read_quadratures.size(); ++j) {
      std::vector<Spectral::Quadrature> temp_quad;
      for (size_t k = 0; k < read_quadratures[j].size(); ++k) {
        temp_quad.push_back(Spectral::to_quadrature(read_quadratures[j][k]));
      }
      vector_of_quadrature.push_back(temp_quad);
    }

    std::cout << read_bases[0].size() << "\n";
    std::cout << read_quadratures.size() << "\n";

    std::vector<ElementVolumeData> element_volume_data_vector;

    for (size_t j = 0; j < read_extents.size(); ++j) {
      element_volume_data_vector.push_back(
          {read_extents[j], tensor_component, vector_of_bases[j],
           vector_of_quadrature[j], read_grid_names[j]});
    }

    auto& new_volume_file = new_file.get<h5::VolumeData>("/" + subfile_prefix);
    new_volume_file.write_volume_data(read_observation_ids[i],
                                      read_observation_value,
                                      element_volume_data_vector);

    new_file.close_current_object();
  }
}

int main(int argc, char** argv) {
  boost::program_options::positional_options_description pos_desc;
  //   pos_desc.add("old_spec_cce_file", 1).add("output_prefix", 1);

  boost::program_options::options_description desc("Options");
  desc.add_options()("help,h,", "show this help message")(
      "file_prefix", boost::program_options::value<std::string>()->required(),
      "prefix of files to be combined")(
      "subfile_prefix",
      boost::program_options::value<std::string>()->required(),
      "shared subfile name in each volume file")(
      "output_prefix", boost::program_options::value<std::string>()->required(),
      "combined output filename");

  boost::program_options::variables_map vars;

  boost::program_options::store(
      boost::program_options::command_line_parser(argc, argv)
          .positional(pos_desc)
          .options(desc)
          .run(),
      vars);

  if (vars.count("help") != 0u or vars.count("file_prefix") == 0u or
      vars.count("subfile_prefix") == 0u or vars.count("output_prefix") == 0u) {
    Parallel::printf("%s\n", desc);
    return 0;
  }

  combine_h5(vars["file_prefix"].as<std::string>(),
             vars["subfile_prefix"].as<std::string>(),
             vars["output_prefix"].as<std::string>());
}

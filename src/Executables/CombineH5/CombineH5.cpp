// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <boost/program_options.hpp>
#include <cstddef>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/VolumeData.hpp"
#include "Parallel/Printf.hpp"

#include <iostream>
#include <vector>

// Charm looks for this function but since we build without a main function or
// main module we just have it be empty
extern "C" void CkRegisterMainModule(void) {}

void combine_h5(const std::string& file_prefix,
                const std::string& output_file) {
  DataVector a{1.0, 2.3, 8.9};
  Parallel::printf("%s\n", a);

  h5::H5File<h5::AccessType::ReadWrite> my_file(file_prefix, true);

  auto& volume_file =
      my_file.get<h5::VolumeData>("/VolumePsiPiPhiEvery50Slabs");

  auto read_observation_ids = volume_file.list_observation_ids();

  for (size_t i = 0; i < read_observation_ids.size(); ++i) {
    std::cout << "ObservationID" << read_observation_ids[i] << ":"
              << "\n";
    std::cout << " - Observation Value: "
              << volume_file.get_observation_value(read_observation_ids[i])
              << "\n";

    auto read_tensor_components =
        volume_file.list_tensor_components(read_observation_ids[i]);
    // for (size_t j = 0; j < read_tensor_components.size(); ++j) {
    //   std::cout << " - Tensor: " << read_tensor_components[j] << "\n";
    //   std::cout << "  * Data: "
    //             << volume_file.get_tensor_component(read_observation_ids[i],
    //                                                read_tensor_components[j])
    //             << "\n";
    // }

    // auto read_bases = volume_file.get_bases(read_observation_ids[i]);
    // std::cout << " - Bases: " << "\n";
    // for (size_t j = 0; j < read_bases.size(); ++j) {
    //     for (size_t k = 0; k < read_bases[j].size(); ++k) {
    //         std::cout << "  * Vector-of-vector-of-STRING??: " <<
    //         read_bases[j][k] << "\n";
    //     }
    // }

    // auto read_quadratures =
    // volume_file.get_quadratures(read_observation_ids[i]); std::cout << " -
    // Quadratures: " << "\n"; for (size_t j = 0; j < read_quadratures.size();
    // ++j) {
    //     for (size_t k = 0; k < read_quadratures[j].size(); ++k) {
    //         std::cout << "  * Vector-of-vector-of-STRING??: " <<
    //         read_quadratures[j][k] << "\n";
    //     }
    // }

    // auto read_extents = volume_file.get_extents(read_observation_ids[i]);
    // std::cout << " - Extents: " << "\n";
    // for (size_t j = 0; j < read_extents.size(); ++j) {
    //     for (size_t k = 0; k < read_extents[j].size(); ++k) {
    //         std::cout << "  * Vector-of-vector-of-nomber??: " <<
    //         read_extents[j][k] << "\n";
    //     }
    // }

    auto read_grid_names = volume_file.get_grid_names(read_observation_ids[i]);
    std::cout << " - Grid Names: "
              << "\n";
    for (size_t j = 0; j < read_grid_names.size(); ++j) {
      std::cout << "  * Vector-of-strings??: " << read_grid_names[j] << "\n";
    }
  }
}

int main(int argc, char** argv) {
  boost::program_options::positional_options_description pos_desc;
  //   pos_desc.add("old_spec_cce_file", 1).add("output_file", 1);

  boost::program_options::options_description desc("Options");
  desc.add_options()("help,h,", "show this help message")(
      "file_prefix", boost::program_options::value<std::string>()->required(),
      "prefix of files to be combined")(
      "output_file", boost::program_options::value<std::string>()->required(),
      "combined output filename");

  boost::program_options::variables_map vars;

  boost::program_options::store(
      boost::program_options::command_line_parser(argc, argv)
          .positional(pos_desc)
          .options(desc)
          .run(),
      vars);

  if (vars.count("help") != 0u or vars.count("file_prefix") == 0u or
      vars.count("output_file") == 0u) {
    Parallel::printf("%s\n", desc);
    return 0;
  }

  combine_h5(vars["file_prefix"].as<std::string>(),
             vars["output_file"].as<std::string>());
}

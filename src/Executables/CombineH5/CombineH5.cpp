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
#include "IO/H5/SourceArchive.hpp"
#include "IO/H5/VolumeData.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/Printf.hpp"
#include "Utilities/FileSystem.hpp"

#include <iostream>

// Charm looks for this function but since we build without a main function or
// main module we just have it be empty
extern "C" void CkRegisterMainModule(void) {}

// Compares source archives of each file against the 0th volume file. Returns
// true if each source archive matches the 0th one
bool check_src_files(const std::vector<std::string> file_names) {
  h5::H5File<h5::AccessType::ReadWrite> initial_file(file_names[0], true);
  auto& src_archive_object_initial =
      initial_file.get<h5::SourceArchive>("/src");
  const std::vector<char>& src_tar_initial =
      src_archive_object_initial.get_archive();
  initial_file.close_current_object();
  for (size_t i = 1; i < file_names.size(); ++i) {
    h5::H5File<h5::AccessType::ReadWrite> comparison_file(file_names[i], true);
    auto& src_archive_object_compare =
        comparison_file.get<h5::SourceArchive>("/src");
    const std::vector<char>& src_tar_compare =
        src_archive_object_compare.get_archive();
    comparison_file.close_current_object();

    if (src_tar_initial != src_tar_compare) {
      return false;
    }
  }
  return true;  // Ask if should have a static_assert() to check that there
                // exists more than one file to combine in the directory
}

void combine_h5(const std::string& file_prefix, const std::string& subfile_name,
                const std::string& output) {
  // Parses for and stores all input files to be looped over
  std::vector<std::string> file_names = file_system::glob(file_prefix + "*.h5");

  // Checks that volume data was generated with identical versions of SpECTRE
  if (check_src_files(file_names) == false) {
    ERROR(
        "One or more of your files were found to have differing src.tar.gz "
        "files, meaning that they may be from differing versions of SpECTRE.");
  }

  // THIS IS FOR SAVING src.tar.gz INFO
  // h5::H5File<h5::AccessType::ReadWrite> initial_file(file_names[0], true);
  // auto& src_archive_object_initial =
  //     initial_file.get<h5::SourceArchive>("/src");
  // const std::vector<char>& src_tar_initial =
  //     src_archive_object_initial.get_archive();
  // initial_file.close_current_object();

  // Instantiates the output file and the .vol subfile to be filled with the
  // combined data later
  h5::H5File<h5::AccessType::ReadWrite> new_file(output + "0.h5", true);
  new_file.insert<h5::VolumeData>("/" + subfile_name + ".vol");
  // new_file_new_file.get<h5::SourceArchive>("/src"); // THIS IS FOR SAVING
  // src.tar.gz INFO
  new_file.close_current_object();

  // std::map to store the sorted volume data
  std::map<int, std::vector<
                    std::tuple<size_t, double, std::vector<ElementVolumeData>>>>
      file_map;

  // Opens the original files, sorts the volume data, and stores it in the
  // std::map
  for (size_t i = 0; i < file_names.size(); ++i) {
    h5::H5File<h5::AccessType::ReadWrite> original_file(
        file_prefix + std::to_string(i) + ".h5", true);

    auto& original_volume_file =
        original_file.get<h5::VolumeData>("/" + subfile_name);

    auto sorted_volume_data = original_volume_file.get_data_by_element(
        std::nullopt, std::nullopt, std::nullopt);

    file_map.insert(
        std::pair<int, std::vector<std::tuple<size_t, double,
                                              std::vector<ElementVolumeData>>>>(
            i, sorted_volume_data));
    original_file.close_current_object();
  }

  // Append vectors of ElementVolumeData that were split between H5 files for
  // a given observation id and value
  for (size_t i = 1; i < file_names.size(); ++i) {
    for (size_t j = 0; j < file_map.at(0).size(); ++j) {
      std::get<2>(file_map.at(0)[j])
          .insert(std::get<2>(file_map.at(0)[j]).end(),
                  std::get<2>(file_map.at(i)[j]).begin(),
                  std::get<2>(file_map.at(i)[j]).end());
    }
  }

  // Write combined volume data for the output file
  auto& new_volume_file = new_file.get<h5::VolumeData>("/" + subfile_name);
  for (auto const& [observation_id, observation_value, element_data] :
       file_map.at(0)) {
    new_volume_file.write_volume_data(observation_id, observation_value,
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

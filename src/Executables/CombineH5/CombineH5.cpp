// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <boost/program_options.hpp>
#include <cstddef>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/VolumeData.hpp"
#include "Parallel/Printf.hpp"

// Charm looks for this function but since we build without a main function or
// main module we just have it be empty
extern "C" void CkRegisterMainModule(void) {}

void combine_h5(const std::string& file_prefix,
                const std::string& output_file) {
  DataVector a{1.0, 2.3, 8.9};
  Parallel::printf("%s\n", a);
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

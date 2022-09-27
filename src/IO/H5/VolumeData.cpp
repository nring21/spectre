// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/H5/VolumeData.hpp"

#include <algorithm>
#include <array>
#include <boost/algorithm/string.hpp>
#include <boost/functional/hash.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <hdf5.h>
#include <memory>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TensorData.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "IO/Connectivity.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/Header.hpp"
#include "IO/H5/Helpers.hpp"
#include "IO/H5/SpectralIo.hpp"
#include "IO/H5/Type.hpp"
#include "IO/H5/Version.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/Printf.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ErrorHandling/ExpectsAndEnsures.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/StdHelpers.hpp"

namespace h5 {
namespace {
// Append the element extents and connectevity to the total extents and
// connectivity
void append_element_extents_and_connectivity(
    const gsl::not_null<std::vector<size_t>*> total_extents,
    const gsl::not_null<std::vector<int>*> total_connectivity,
    const gsl::not_null<std::vector<int>*> pole_connectivity,
    const gsl::not_null<int*> total_points_so_far, const size_t dim,
    const ElementVolumeData& element) {
  // Process the element extents
  const auto& extents = element.extents;
  ASSERT(alg::none_of(extents, [](const size_t extent) { return extent == 1; }),
         "We cannot generate connectivity for any single grid point elements.");
  if (extents.size() != dim) {
    ERROR("Trying to write data of dimensionality"
          << extents.size() << "but the VolumeData file has dimensionality"
          << dim << ".");
  }
  total_extents->insert(total_extents->end(), extents.begin(), extents.end());
  // Find the number of points in the local connectivity
  const int element_num_points =
      alg::accumulate(extents, 1, std::multiplies<>{});
  // Generate the connectivity data for the element
  // Possible optimization: local_connectivity.reserve(BLAH) if we can figure
  // out size without computing all the connectivities.
  const std::vector<int> connectivity = [&extents, &total_points_so_far]() {
    std::vector<int> local_connectivity;
    for (const auto& cell : vis::detail::compute_cells(extents)) {
      for (const auto& bounding_indices : cell.bounding_indices) {
        local_connectivity.emplace_back(*total_points_so_far +
                                        static_cast<int>(bounding_indices));
      }
    }
    return local_connectivity;
  }();
  *total_points_so_far += element_num_points;
  total_connectivity->insert(total_connectivity->end(), connectivity.begin(),
                             connectivity.end());

  // If element is 2D and the bases are both SphericalHarmonic,
  // then add extra connections to close the surface.
  if (dim == 2) {
    if (element.basis[0] == Spectral::Basis::SphericalHarmonic and
        element.basis[1] == Spectral::Basis::SphericalHarmonic) {
      // Extents are (l+1, 2l+1)
      const size_t l = element.extents[0] - 1;

      // Connect max(phi) and min(phi) by adding more quads
      // to total_connectivity
      for (size_t j = 0; j < l; ++j) {
        total_connectivity->push_back(j);
        total_connectivity->push_back(j + 1);
        total_connectivity->push_back(2 * l * (l + 1) + j + 1);
        total_connectivity->push_back((2 * l) * (l + 1) + j);
      }

      // Add a new connectivity output for filling the poles
      // First, get the points at min(theta), which define the
      // boundary of the top pole to fill, and the points at
      // max(theta), which define the boundary of the bottom
      // pole to fill. Note: points are stored with theta
      // varying faster than phi.
      std::vector<int> top_pole_points{};
      std::vector<int> bottom_pole_points{};
      for (size_t k = 0; k < (2 * l + 1); ++k) {
        top_pole_points.push_back(k * (l + 1));
        bottom_pole_points.push_back(k * (l + 1) + l);
      }

      // Fill the poles with triangles. Start by connecting
      // points 0,1,2, 2,3,4, etc. into small triangles,
      // then connect points 0,2,4, 4,6,8, etc.,
      // etc., until fewer than 3 points remain.
      const size_t number_of_points_near_poles = top_pole_points.size();
      size_t to_next_triangle_point = 1;
      while (number_of_points_near_poles / to_next_triangle_point >= 3) {
        for (size_t point_starting_triangle = 0;
             point_starting_triangle <
             number_of_points_near_poles - 2 * to_next_triangle_point;
             point_starting_triangle += 2 * to_next_triangle_point) {
          pole_connectivity->push_back(
              gsl::at(top_pole_points, point_starting_triangle));
          pole_connectivity->push_back(
              gsl::at(top_pole_points,
                      point_starting_triangle + to_next_triangle_point));
          pole_connectivity->push_back(
              gsl::at(top_pole_points,
                      point_starting_triangle + 2 * to_next_triangle_point));
          pole_connectivity->push_back(
              gsl::at(bottom_pole_points, point_starting_triangle));
          pole_connectivity->push_back(
              gsl::at(bottom_pole_points,
                      point_starting_triangle + to_next_triangle_point));
          pole_connectivity->push_back(
              gsl::at(bottom_pole_points,
                      point_starting_triangle + 2 * to_next_triangle_point));
        }
        // If odd number of points, add triangle closing
        // point at max(phi) and point at min(phi)
        if (number_of_points_near_poles % 2 != 0 and
            2 * to_next_triangle_point < number_of_points_near_poles) {
          pole_connectivity->push_back(gsl::at(
              top_pole_points,
              number_of_points_near_poles - 2 * to_next_triangle_point));
          pole_connectivity->push_back(
              gsl::at(top_pole_points,
                      number_of_points_near_poles - to_next_triangle_point));
          pole_connectivity->push_back(gsl::at(top_pole_points, 0));
          pole_connectivity->push_back(gsl::at(
              bottom_pole_points,
              number_of_points_near_poles - 2 * to_next_triangle_point));
          pole_connectivity->push_back(
              gsl::at(bottom_pole_points,
                      number_of_points_near_poles - to_next_triangle_point));
          pole_connectivity->push_back(gsl::at(bottom_pole_points, 0));
        }
        to_next_triangle_point += 1;
      }
    }
  }
}

// Given a std::vector of grid_names, computes the number of blocks that exist
// and also returns a std::vector of block numbers that is a one-to-one mapping
// to each element in grid_names
std::tuple<size_t, std::vector<size_t>, std::vector<std::vector<size_t>>>
compute_and_organize_block_info(std::vector<std::string>& grid_names) {
  std::vector<size_t> block_number_for_each_element;
  std::vector<std::vector<size_t>> sorted_element_indices;
  block_number_for_each_element.reserve(grid_names.size());

  // Fills block_number_for_each_element
  for (size_t i = 0; i < grid_names.size(); ++i) {
    size_t end_position = grid_names[i].find(",", 1);
    block_number_for_each_element.push_back(
        static_cast<size_t>(std::stoi(grid_names[i].substr(2, end_position))));
  }

  auto max_block_number =
      *std::max_element(block_number_for_each_element.begin(),
                        block_number_for_each_element.end());
  auto number_of_blocks = max_block_number + 1;
  sorted_element_indices.reserve(number_of_blocks);

  // Properly sizes subvectors of sorted_element_indices
  for (size_t i = 0; i < number_of_blocks; ++i) {
    std::vector<size_t> sizing_vector;
    auto grid_names_in_block =
        static_cast<size_t>(std::count(block_number_for_each_element.begin(),
                                       block_number_for_each_element.end(), i));
    sizing_vector.reserve(grid_names_in_block);
    sorted_element_indices.push_back(sizing_vector);
  }

  // Organizing grid_names by block
  for (size_t i = 0; i < block_number_for_each_element.size(); ++i) {
    sorted_element_indices[block_number_for_each_element[i]].push_back(i);
  }

  return std::make_tuple(number_of_blocks, block_number_for_each_element,
                         sorted_element_indices);
}

// Sort input by block
template <typename T>
std::vector<std::vector<T>> sort_by_block(
    const std::vector<std::vector<size_t>>& sorted_element_indices,
    const std::vector<T>& property_to_sort) {
  std::vector<std::vector<T>> sorted_property;
  sorted_property.reserve(sorted_element_indices.size());

  // Properly sizes subvectors
  for (size_t i = 0; i < sorted_element_indices.size(); ++i) {
    std::vector<T> sizing_vector;
    sizing_vector.reserve(sorted_element_indices[i].size());
    for (size_t j = 0; j < sorted_element_indices[i].size(); ++j) {
      sizing_vector.push_back(property_to_sort[sorted_element_indices[i][j]]);
    }
    sorted_property.push_back(sizing_vector);
  }

  return sorted_property;
}

// Returns properties for each block
std::tuple<size_t, size_t, std::array<int, 3>> compute_block_level_properties(
    const std::vector<std::string>& block_grid_names,
    const std::vector<std::vector<size_t>>& block_extents) {
  size_t expected_connectivity_length = 0;
  size_t expected_number_of_grid_points = 0;

  for (size_t i = 0; i < block_extents.size(); ++i) {
    // Connectivity that already exists
    expected_connectivity_length += (block_extents[i][0] - 1) *
                                    (block_extents[i][1] - 1) *
                                    (block_extents[i][2] - 1) * 8;
    // Used for reserving the length of block_logical_coords
    expected_number_of_grid_points +=
        block_extents[i][0] * block_extents[i][1] * block_extents[i][2];
  }

  std::string grid_name_string = block_grid_names[0];
  std::array<int, 3> h_ref_array;
  size_t h_ref_previous_start_position = 0;
  for (size_t i = 0; i < 3; ++i) {
    size_t h_ref_start_position =
        grid_name_string.find("L", h_ref_previous_start_position + 1);
    size_t h_ref_end_position =
        grid_name_string.find("I", h_ref_start_position);
    int h_ref = std::stoi(
        grid_name_string.substr(h_ref_start_position + 1,
                                h_ref_end_position - h_ref_start_position - 1));
    h_ref_array[i] = h_ref;
    h_ref_previous_start_position = h_ref_start_position;
  }
  expected_connectivity_length +=
      ((pow(2, h_ref_array[0] + 1) - 1) * (pow(2, h_ref_array[1] + 1) - 1) *
           (pow(2, h_ref_array[2] + 1) - 1) -
       block_extents.size()) *
      8;

  return std::tuple{expected_connectivity_length,
                    expected_number_of_grid_points, h_ref_array};
}

// Generates the mesh
Mesh<3> generate_element_mesh(
    const std::vector<std::string>& element_bases,
    const std::vector<std::string>& element_quadratures,
    const std::vector<size_t>& element_extents) {
  std::array<Spectral::Basis, 3> basis_array = {
      Spectral::to_basis(element_bases[0]),
      Spectral::to_basis(element_bases[1]),
      Spectral::to_basis(element_bases[2])};

  std::array<Spectral::Quadrature, 3> quadrature_array = {
      Spectral::to_quadrature(element_quadratures[0]),
      Spectral::to_quadrature(element_quadratures[1]),
      Spectral::to_quadrature(element_quadratures[2])};

  std::array<size_t, 3> extents_array = {element_extents[0], element_extents[1],
                                         element_extents[2]};

  return Mesh<3>{extents_array, basis_array, quadrature_array};
}
std::vector<std::array<double, 3>> generate_block_logical_coordinates(
    const std::vector<std::array<double, 3>>& element_logical_coordinates,
    const std::string& grid_name,
    const std::array<int, 3>& h_refinement_array) {
  size_t grid_points_x_start_position = 0;
  std::vector<std::array<double, 3>> block_logical_coordinates;
  block_logical_coordinates.reserve(element_logical_coordinates.size());
  std::vector<double> number_of_elements_each_direction;
  number_of_elements_each_direction.reserve(3);  // Hardcoded for 3 dim
  std::vector<double> shift_each_direction;
  shift_each_direction.reserve(3);  // Hardocded for 3 dim

  for (size_t i = 0; i < 3; ++i) {
    double number_of_elements = pow(2, h_refinement_array[i]);
    number_of_elements_each_direction.push_back(number_of_elements);
    size_t grid_points_start_position =
        grid_name.find("I", grid_points_x_start_position + 1);
    size_t grid_points_end_position =
        grid_name.find(",", grid_points_start_position);
    if (i == 2) {
      grid_points_end_position =
          grid_name.find(")", grid_points_start_position);
    }
    int element_index = std::stoi(grid_name.substr(
        grid_points_start_position + 1,
        grid_points_end_position - grid_points_start_position - 1));
    double shift = (-1 + (2 * element_index + 1) / number_of_elements);
    shift_each_direction.push_back(shift);
    grid_points_x_start_position = grid_points_start_position;
  }

  for (size_t i = 0; i < element_logical_coordinates.size(); ++i) {
    std::array<double, 3> grid_point_coordinate;
    for (size_t j = 0; j < grid_point_coordinate.size(); ++j) {
      grid_point_coordinate[j] = 1 / number_of_elements_each_direction[j] *
                                     element_logical_coordinates[i][j] +
                                 shift_each_direction[j];
    }
    block_logical_coordinates.push_back(grid_point_coordinate);
  }

  return block_logical_coordinates;
}

// Sorting routine for an incoming list of values
std::vector<double> sort_and_order(std::vector<double>& unsorted_coordinate) {
  std::vector<double> sorted_coordinate;
  sort(unsorted_coordinate.begin(), unsorted_coordinate.end());
  sorted_coordinate.push_back(unsorted_coordinate[0]);
  for (size_t i = 1; i < unsorted_coordinate.size(); ++i) {
    if (unsorted_coordinate[i] == sorted_coordinate.end()[-1]) {
      continue;
    } else {
      sorted_coordinate.push_back(unsorted_coordinate[i]);
    }
  }
  return sorted_coordinate;
}

// Builds the connectivity by cube
std::vector<std::pair<size_t, std::array<double, 3>>>
build_connectivity_by_hexahedron(std::vector<double>& sorted_x,
                                 std::vector<double>& sorted_y,
                                 std::vector<double>& sorted_z,
                                 size_t& block_number) {
  std::vector<std::pair<size_t, std::array<double, 3>>> connectivity_of_keys;

  for (size_t k = 0; k < sorted_z.size() - 1; ++k) {
    for (size_t j = 0; j < sorted_y.size() - 1; ++j) {
      for (size_t i = 0; i < sorted_x.size() - 1; ++i) {
        std::array point_one{sorted_x[i], sorted_y[j], sorted_z[k]};
        std::array point_two{sorted_x[i + 1], sorted_y[j], sorted_z[k]};
        std::array point_three{sorted_x[i + 1], sorted_y[j + 1], sorted_z[k]};
        std::array point_four{sorted_x[i], sorted_y[j + 1], sorted_z[k]};
        std::array point_five{sorted_x[i], sorted_y[j], sorted_z[k + 1]};
        std::array point_six{sorted_x[i + 1], sorted_y[j], sorted_z[k + 1]};
        std::array point_seven{sorted_x[i + 1], sorted_y[j + 1],
                               sorted_z[k + 1]};
        std::array point_eight{sorted_x[i], sorted_y[j + 1], sorted_z[k + 1]};
        connectivity_of_keys.insert(
            connectivity_of_keys.end(),
            {std::make_pair(block_number, point_one),
             std::make_pair(block_number, point_two),
             std::make_pair(block_number, point_three),
             std::make_pair(block_number, point_four),
             std::make_pair(block_number, point_five),
             std::make_pair(block_number, point_six),
             std::make_pair(block_number, point_seven),
             std::make_pair(block_number, point_eight)});
      }
    }
  }
  return connectivity_of_keys;
}

// Generates the new connectivity
std::vector<std::pair<size_t, std::array<double, 3>>> generate_new_connectivity(
    std::vector<std::array<double, 3>>& block_logical_coordinates,
    size_t& block_number) {
  std::vector<std::vector<double>> unsorted_coordinates;
  unsorted_coordinates.reserve(3);  // Hardcoded for dim 3

  for (size_t i = 0; i < 3; ++i) {
    std::vector<double> coordinates_by_direction;
    coordinates_by_direction.reserve(block_logical_coordinates.size());
    for (size_t j = 0; j < block_logical_coordinates.size(); ++j) {
      coordinates_by_direction.push_back(block_logical_coordinates[j][i]);
    }
    unsorted_coordinates.push_back(coordinates_by_direction);
  }

  auto ordered_x = sort_and_order(unsorted_coordinates[0]);
  auto ordered_y = sort_and_order(unsorted_coordinates[1]);
  auto ordered_z = sort_and_order(unsorted_coordinates[2]);
  return build_connectivity_by_hexahedron(ordered_x, ordered_y, ordered_z,
                                          block_number);
}
}  // namespace

VolumeData::VolumeData(const bool subfile_exists, detail::OpenGroup&& group,
                       const hid_t /*location*/, const std::string& name,
                       const uint32_t version)
    : group_(std::move(group)),
      name_(name.size() > extension().size()
                ? (extension() == name.substr(name.size() - extension().size())
                       ? name
                       : name + extension())
                : name + extension()),
      path_(group_.group_path_with_trailing_slash() + name),
      version_(version),
      volume_data_group_(group_.id(), name_, h5::AccessType::ReadWrite) {
  if (subfile_exists) {
    // We treat this as an internal version for now. We'll need to deal with
    // proper versioning later.
    const Version open_version(true, detail::OpenGroup{},
                               volume_data_group_.id(), "version");
    version_ = open_version.get_version();
    const Header header(true, detail::OpenGroup{}, volume_data_group_.id(),
                        "header");
    header_ = header.get_header();
  } else {  // file does not exist
    // Subfiles are closed as they go out of scope, so we have the extra
    // braces here to add the necessary scope
    {
      Version open_version(false, detail::OpenGroup{}, volume_data_group_.id(),
                           "version", version_);
    }
    {
      Header header(false, detail::OpenGroup{}, volume_data_group_.id(),
                    "header");
      header_ = header.get_header();
    }
  }
}

// Write Volume Data stored in a vector of `ExtentsAndTensorVolumeData` to
// an `observation_group` in a `VolumeData` file.
void VolumeData::write_volume_data(
    const size_t observation_id, const double observation_value,
    const std::vector<ElementVolumeData>& elements) {
  const std::string path = "ObservationId" + std::to_string(observation_id);
  detail::OpenGroup observation_group(volume_data_group_.id(), path,
                                      AccessType::ReadWrite);
  if (contains_attribute(observation_group.id(), "", "observation_value")) {
    ERROR_NO_TRACE("Trying to write ObservationId "
                   << std::to_string(observation_id)
                   << " with observation_value " << observation_group.id()
                   << " which already exists in file at " << path
                   << ". Did you forget to clean up after an earlier run?");
  }
  h5::write_to_attribute(observation_group.id(), "observation_value",
                         observation_value);
  // Get first element to extract the component names and dimension
  const auto get_component_name = [](const auto& component) {
    ASSERT(component.name.find_last_of('/') == std::string::npos,
           "The expected format of the tensor component names is "
           "'COMPONENT_NAME' but found a '/' in '"
               << component.name << "'.");
    return component.name;
  };
  const std::vector<std::string> component_names(
      boost::make_transform_iterator(elements.front().tensor_components.begin(),
                                     get_component_name),
      boost::make_transform_iterator(elements.front().tensor_components.end(),
                                     get_component_name));
  // The dimension of the grid is the number of extents per element. I.e., if
  // the extents are [8,5,7] for any element, the dimension of the grid is 3.
  // Only written once per VolumeData file (All volume data in a single file
  // should have the same dimensionality)
  if (not contains_attribute(volume_data_group_.id(), "", "dimension")) {
    h5::write_to_attribute(volume_data_group_.id(), "dimension",
                           elements.front().extents.size());
  }
  const auto dim =
      h5::read_value_attribute<size_t>(volume_data_group_.id(), "dimension");
  // Extract Tensor Data one component at a time
  std::vector<size_t> total_extents;
  std::string grid_names;
  std::vector<int> total_connectivity;
  std::vector<int> pole_connectivity{};
  std::vector<int> quadratures;
  std::vector<int> bases;
  // Keep a running count of the number of points so far to use as a global
  // index for the connectivity
  int total_points_so_far = 0;
  // Loop over tensor components
  for (size_t i = 0; i < component_names.size(); i++) {
    std::string component_name = component_names[i];
    // Write the data for the tensor component
    if (h5::contains_dataset_or_group(observation_group.id(), "",
                                      component_name)) {
      ERROR("Trying to write tensor component '"
            << component_name
            << "' which already exists in HDF5 file in group '" << name_ << '/'
            << "ObservationId" << std::to_string(observation_id) << "'");
    }

    const auto fill_and_write_contiguous_tensor_data =
        [&bases, &component_name, &dim, &elements, &grid_names, i,
         &observation_group, &quadratures, &total_connectivity,
         &pole_connectivity, &total_extents,
         &total_points_so_far](const auto contiguous_tensor_data_ptr) {
          for (const auto& element : elements) {
            if (UNLIKELY(i == 0)) {
              // True if first tensor component being accessed
              grid_names += element.element_name + h5::VolumeData::separator();
              // append element basis
              alg::transform(
                  element.basis, std::back_inserter(bases),
                  [](const Spectral::Basis t) { return static_cast<int>(t); });
              // append element quadraature
              alg::transform(element.quadrature,
                             std::back_inserter(quadratures),
                             [](const Spectral::Quadrature t) {
                               return static_cast<int>(t);
                             });

              append_element_extents_and_connectivity(
                  &total_extents, &total_connectivity, &pole_connectivity,
                  &total_points_so_far, dim, element);
            }
            using type_from_variant = tmpl::conditional_t<
                std::is_same_v<
                    std::decay_t<decltype(*contiguous_tensor_data_ptr)>,
                    std::vector<double>>,
                DataVector, std::vector<float>>;
            contiguous_tensor_data_ptr->insert(
                contiguous_tensor_data_ptr->end(),
                std::get<type_from_variant>(element.tensor_components[i].data)
                    .begin(),
                std::get<type_from_variant>(element.tensor_components[i].data)
                    .end());
          }  // for each element
          h5::write_data(observation_group.id(), *contiguous_tensor_data_ptr,
                         {contiguous_tensor_data_ptr->size()}, component_name);
        };

    if (elements[0].tensor_components[i].data.index() == 0) {
      std::vector<double> contiguous_tensor_data{};
      fill_and_write_contiguous_tensor_data(
          make_not_null(&contiguous_tensor_data));
    } else if (elements[0].tensor_components[i].data.index() == 1) {
      std::vector<float> contiguous_tensor_data{};
      fill_and_write_contiguous_tensor_data(
          make_not_null(&contiguous_tensor_data));
    } else {
      ERROR("Unknown index value ("
            << elements[0].tensor_components[i].data.index()
            << ") in std::variant of tensor component.");
    }
  }  // for each component
  grid_names.pop_back();

  // Write the grid extents contiguously, the first `dim` belong to the
  // First grid, the second `dim` belong to the second grid, and so on,
  // Ordering is `x, y, z, ... `
  h5::write_data(observation_group.id(), total_extents, {total_extents.size()},
                 "total_extents");
  // Write the names of the grids as vector of chars with individual names
  // separated by `separator()`
  std::vector<char> grid_names_as_chars(grid_names.begin(), grid_names.end());
  h5::write_data(observation_group.id(), grid_names_as_chars,
                 {grid_names_as_chars.size()}, "grid_names");
  // Write the coded quadrature, along with the dictionary
  const auto io_quadratures = h5_detail::allowed_quadratures();
  std::vector<std::string> quadrature_dict(io_quadratures.size());
  alg::transform(io_quadratures, quadrature_dict.begin(),
                 get_output<Spectral::Quadrature>);
  h5_detail::write_dictionary("Quadrature dictionary", quadrature_dict,
                              observation_group);
  h5::write_data(observation_group.id(), quadratures, {quadratures.size()},
                 "quadratures");
  // Write the coded basis, along with the dictionary
  const auto io_bases = h5_detail::allowed_bases();
  std::vector<std::string> basis_dict(io_bases.size());
  alg::transform(io_bases, basis_dict.begin(), get_output<Spectral::Basis>);
  h5_detail::write_dictionary("Basis dictionary", basis_dict,
                              observation_group);
  h5::write_data(observation_group.id(), bases, {bases.size()}, "bases");
  // Write the Connectivity
  h5::write_data(observation_group.id(), total_connectivity,
                 {total_connectivity.size()}, "connectivity");
  // Note: pole_connectivity stores extra connections that define triangles to
  // fill in the poles on a Strahlkorper and is empty if not outputting
  // Strahlkorper surface data. Because these connections define triangles
  // and not quadrilaterals, they are stored separately instead of just being
  // included in total_connectivity.
  if (not pole_connectivity.empty()) {
    h5::write_data(observation_group.id(), pole_connectivity,
                   {pole_connectivity.size()}, "pole_connectivity");
  }
}

// Write new connectivity connections given a std::vector of observation ids
void VolumeData::write_new_connectivity_data(
    const std::vector<size_t>& observation_ids, const bool& print_size) {
  for (size_t i = 0; i < observation_ids.size(); ++i) {
    auto obs_id = observation_ids[i];
    auto grid_names = get_grid_names(obs_id);
    auto extents = get_extents(obs_id);
    auto bases = get_bases(obs_id);
    auto quadratures = get_quadratures(obs_id);

    auto [number_of_blocks, block_number_for_each_element,
          sorted_element_indices] = compute_and_organize_block_info(grid_names);

    auto sorted_grid_names = sort_by_block(sorted_element_indices, grid_names);
    auto sorted_extents = sort_by_block(sorted_element_indices, extents);

    size_t total_expected_connectivity = 0;
    std::vector<int> expected_grid_points_per_block;
    expected_grid_points_per_block.reserve(number_of_blocks);
    std::vector<std::array<int, 3>> h_ref_per_block;
    h_ref_per_block.reserve(number_of_blocks);

    // Loop over blocks
    for (size_t j = 0; j < number_of_blocks; ++j) {
      auto [expected_connectivity_length, expected_number_of_grid_points,
            h_ref_array] = compute_block_level_properties(sorted_grid_names[j],
                                                          sorted_extents[j]);
      total_expected_connectivity += expected_connectivity_length;
      expected_grid_points_per_block.push_back(expected_number_of_grid_points);
      h_ref_per_block.push_back(h_ref_array);
    }

    // Instantiates the unordered_map
    std::unordered_map<std::pair<size_t, std::array<double, 3>>, size_t,
                       boost::hash<std::pair<size_t, std::array<double, 3>>>>
        block_and_grid_point_map;

    // Instantiates the sorted container for the grid points
    std::vector<std::vector<std::array<double, 3>>>
        block_logical_coordinates_by_block;
    block_logical_coordinates_by_block.reserve(number_of_blocks);

    // Reserve size for the subvectors
    for (size_t j = 0; j < sorted_element_indices.size(); ++j) {
      std::vector<std::array<double, 3>> sizing_vector;
      sizing_vector.reserve(sorted_element_indices[j].size());
      block_logical_coordinates_by_block.push_back(sizing_vector);
    }

    // Counter for the unordered_map
    size_t grid_point_counter = 0;

    // Loop over elements
    for (size_t j = 0; j < block_number_for_each_element.size(); ++j) {
      auto element_mesh =
          generate_element_mesh(bases[j], quadratures[j], extents[j]);
      auto element_logical_coordinates_tensor =
          logical_coordinates(element_mesh);

      std::vector<std::array<double, 3>> element_logical_coordinates;
      element_logical_coordinates.reserve(
          element_logical_coordinates_tensor.get(0).size());

      for (size_t k = 0; k < element_logical_coordinates_tensor.get(0).size();
           ++k) {
        element_logical_coordinates.push_back(std::array<double, 3>{
            element_logical_coordinates_tensor.get(0)[k],
            element_logical_coordinates_tensor.get(1)[k],
            element_logical_coordinates_tensor.get(2)[k]});
      }

      auto block_logical_coordinates = generate_block_logical_coordinates(
          element_logical_coordinates, grid_names[j],
          h_ref_per_block[block_number_for_each_element[j]]);

      // Stores (B#, grid_point_coord_array) -> grid_point_number in an
      // unordered_map and grid_point_coord_array by block
      for (size_t k = 0; k < block_logical_coordinates.size(); ++k) {
        std::pair<size_t, std::array<double, 3>> block_and_grid_point(
            block_number_for_each_element[j], block_logical_coordinates[k]);
        block_and_grid_point_map.insert(
            std::pair<std::pair<size_t, std::array<double, 3>>, size_t>(
                block_and_grid_point, grid_point_counter));
        grid_point_counter += 1;

        block_logical_coordinates_by_block[block_number_for_each_element[j]]
            .push_back(block_logical_coordinates[k]);
      }
    }

    std::vector<int> new_connectivity;
    new_connectivity.reserve(total_expected_connectivity);

    for (size_t j = 0; j < block_logical_coordinates_by_block.size(); ++j) {
      auto block_number = j;
      auto connectivity_of_keys = generate_new_connectivity(
          block_logical_coordinates_by_block[j], block_number);
      for (const std::pair<size_t, std::array<double, 3>>& k :
           connectivity_of_keys) {
        new_connectivity.push_back(block_and_grid_point_map[k]);
      }
    }

    // Deletes the existing connectivity and replaces it with the new one
    const std::string path = "ObservationId" + std::to_string(obs_id);
    detail::OpenGroup observation_group(volume_data_group_.id(), path,
                                        AccessType::ReadWrite);
    const hid_t group_id = observation_group.id();
    H5Ldelete(group_id, "connectivity", H5P_DEFAULT);
    write_connectivity(group_id, new_connectivity);
    if (print_size == true) {
      Parallel::printf("connectivity length: %i\n", new_connectivity.size());
    }
  }
}

std::vector<size_t> VolumeData::list_observation_ids() const {
  const auto names = get_group_names(volume_data_group_.id(), "");
  const auto helper = [](const std::string& s) {
    return std::stoul(s.substr(std::string("ObservationId").size()));
  };
  std::vector<size_t> obs_ids{
      boost::make_transform_iterator(names.begin(), helper),
      boost::make_transform_iterator(names.end(), helper)};
  alg::sort(obs_ids, [this](const size_t lhs, const size_t rhs) {
    return this->get_observation_value(lhs) < this->get_observation_value(rhs);
  });
  return obs_ids;
}

double VolumeData::get_observation_value(const size_t observation_id) const {
  const std::string path = "ObservationId" + std::to_string(observation_id);
  detail::OpenGroup observation_group(volume_data_group_.id(), path,
                                      AccessType::ReadOnly);
  return h5::read_value_attribute<double>(observation_group.id(),
                                          "observation_value");
}

std::vector<std::string> VolumeData::list_tensor_components(
    const size_t observation_id) const {
  auto tensor_components =
      get_group_names(volume_data_group_.id(),
                      "ObservationId" + std::to_string(observation_id));
  auto remove_data_name = [&tensor_components](const std::string& data_name) {
    // NOLINTNEXTLINE(bugprone-unused-return-value)
    alg::remove(tensor_components, data_name);
  };

  const auto search_result = alg::find(tensor_components, "pole_connectivity");
  const long number_of_components_to_remove =
      search_result != tensor_components.end() ? 6 : 5;

  remove_data_name("connectivity");
  remove_data_name("pole_connectivity");
  remove_data_name("total_extents");
  remove_data_name("grid_names");
  remove_data_name("quadratures");
  remove_data_name("bases");
  // std::remove moves the element to the end of the vector, so we still need to
  // actually erase it from the vector
  tensor_components.erase(
      tensor_components.end() - number_of_components_to_remove,
      tensor_components.end());

  return tensor_components;
}

std::vector<std::string> VolumeData::get_grid_names(
    const size_t observation_id) const {
  const std::string path = "ObservationId" + std::to_string(observation_id);
  detail::OpenGroup observation_group(volume_data_group_.id(), path,
                                      AccessType::ReadOnly);
  const std::vector<char> names =
      h5::read_data<1, std::vector<char>>(observation_group.id(), "grid_names");
  const std::string all_names(names.begin(), names.end());
  std::vector<std::string> grid_names{};
  boost::split(grid_names, all_names,
               [](const char c) { return c == h5::VolumeData::separator(); });
  return grid_names;
}

TensorComponent VolumeData::get_tensor_component(
    const size_t observation_id, const std::string& tensor_component) const {
  const std::string path = "ObservationId" + std::to_string(observation_id);
  detail::OpenGroup observation_group(volume_data_group_.id(), path,
                                      AccessType::ReadOnly);

  const hid_t dataset_id =
      h5::open_dataset(observation_group.id(), tensor_component);
  const hid_t dataspace_id = h5::open_dataspace(dataset_id);
  const auto rank =
      static_cast<size_t>(H5Sget_simple_extent_ndims(dataspace_id));
  h5::close_dataspace(dataspace_id);
  const bool use_float =
      h5::types_equal(H5Dget_type(dataset_id), h5::h5_type<float>());
  h5::close_dataset(dataset_id);

  const auto get_data = [&observation_group, &rank,
                         &tensor_component](auto type_to_get_v) {
    using type_to_get = tmpl::type_from<decltype(type_to_get_v)>;
    switch (rank) {
      case 1:
        return h5::read_data<1, type_to_get>(observation_group.id(),
                                             tensor_component);
      case 2:
        return h5::read_data<2, type_to_get>(observation_group.id(),
                                             tensor_component);
      case 3:
        return h5::read_data<3, type_to_get>(observation_group.id(),
                                             tensor_component);
      default:
        ERROR("Rank must be 1, 2, or 3. Received data with Rank = " << rank);
    }
  };

  if (use_float) {
    return {tensor_component, get_data(tmpl::type_<std::vector<float>>{})};
  } else {
    return {tensor_component, get_data(tmpl::type_<DataVector>{})};
  }
}

std::vector<std::vector<size_t>> VolumeData::get_extents(
    const size_t observation_id) const {
  const std::string path = "ObservationId" + std::to_string(observation_id);
  detail::OpenGroup observation_group(volume_data_group_.id(), path,
                                      AccessType::ReadOnly);
  const auto dim =
      h5::read_value_attribute<size_t>(volume_data_group_.id(), "dimension");
  const auto extents_per_element = static_cast<long>(dim);
  const auto total_extents = h5::read_data<1, std::vector<size_t>>(
      observation_group.id(), "total_extents");
  std::vector<std::vector<size_t>> individual_extents;
  individual_extents.reserve(total_extents.size() / dim);
  for (auto iter = total_extents.begin(); iter != total_extents.end();
       iter += extents_per_element) {
    individual_extents.emplace_back(iter, iter + extents_per_element);
  }
  return individual_extents;
}

std::pair<size_t, size_t> offset_and_length_for_grid(
    const std::string& grid_name,
    const std::vector<std::string>& all_grid_names,
    const std::vector<std::vector<size_t>>& all_extents) {
  auto found_grid_name = alg::find(all_grid_names, grid_name);
  if (found_grid_name == all_grid_names.end()) {
    ERROR("Found no grid named '" + grid_name + "'.");
  } else {
    const auto element_index =
        std::distance(all_grid_names.begin(), found_grid_name);
    const size_t element_data_offset = std::accumulate(
        all_extents.begin(), all_extents.begin() + element_index, 0_st,
        [](const size_t offset, const std::vector<size_t>& extents) {
          return offset + alg::accumulate(extents, 1_st, std::multiplies<>{});
        });
    const size_t element_data_length = alg::accumulate(
        gsl::at(all_extents, element_index), 1_st, std::multiplies<>{});
    return {element_data_offset, element_data_length};
  }
}

auto VolumeData::get_data_by_element(
    const std::optional<double> start_observation_value,
    const std::optional<double> end_observation_value,
    const std::optional<std::vector<std::string>>& components_to_retrieve) const
    -> std::vector<std::tuple<size_t, double, std::vector<ElementVolumeData>>> {
  // First get list of all observations we need to retrieve
  const auto names = get_group_names(volume_data_group_.id(), "");
  const auto get_observation_id_from_group_name = [](const std::string& s) {
    return std::stoul(s.substr(std::string("ObservationId").size()));
  };
  std::vector<size_t> obs_ids{
      boost::make_transform_iterator(names.begin(),
                                     get_observation_id_from_group_name),
      boost::make_transform_iterator(names.end(),
                                     get_observation_id_from_group_name)};
  std::vector<std::tuple<size_t, double, std::vector<ElementVolumeData>>>
      result{};
  result.reserve(obs_ids.size());
  // Sort observation IDs and observation values into the result. This only
  // copies observed times in
  // [`start_observation_value`, `end_observation_value`]
  for (const auto& observation_id : obs_ids) {
    const double observation_value = get_observation_value(observation_id);
    if (start_observation_value.value_or(
            std::numeric_limits<double>::lowest()) <= observation_value and
        observation_value <= end_observation_value.value_or(
                                 std::numeric_limits<double>::max())) {
      result.emplace_back(observation_id, observation_value,
                          std::vector<ElementVolumeData>{});
    }
  }
  result.shrink_to_fit();
  // Sort by observation_value
  alg::sort(result, [](const auto& lhs, const auto& rhs) {
    return std::get<1>(lhs) < std::get<1>(rhs);
  });

  // Retrieve element data and insert into result
  for (auto& single_time_data : result) {
    const auto known_components =
        list_tensor_components(std::get<0>(single_time_data));

    std::vector<ElementVolumeData> element_volume_data{};
    const auto grid_names = get_grid_names(std::get<0>(single_time_data));
    const auto extents = get_extents(std::get<0>(single_time_data));
    const auto bases = get_bases(std::get<0>(single_time_data));
    const auto quadratures = get_quadratures(std::get<0>(single_time_data));
    element_volume_data.reserve(grid_names.size());

    const auto& component_names =
        components_to_retrieve.value_or(known_components);
    std::vector<TensorComponent> tensors{};
    tensors.reserve(grid_names.size());
    for (const std::string& component : component_names) {
      if (not alg::found(known_components, component)) {
        using ::operator<<;  // STL streams
        ERROR("Could not find tensor component '"
              << component
              << "' in file. Known components are: " << known_components);
      }
      tensors.emplace_back(
          get_tensor_component(std::get<0>(single_time_data), component));
    }
    // Now split the data by element
    for (size_t grid_index = 0, offset = 0; grid_index < grid_names.size();
         ++grid_index) {
      const size_t mesh_size =
          alg::accumulate(extents[grid_index], 1_st, std::multiplies<>{});
      std::vector<TensorComponent> tensor_components{tensors.size()};
      for (size_t component_index = 0; component_index < tensors.size();
           ++component_index) {
        std::visit(
            [component_index, &component_names, mesh_size, offset,
             &tensor_components](const auto& tensor_component_data) {
              std::decay_t<decltype(tensor_component_data)> component(
                  mesh_size);
              std::copy(
                  std::next(tensor_component_data.begin(),
                            static_cast<std::ptrdiff_t>(offset)),
                  std::next(tensor_component_data.begin(),
                            static_cast<std::ptrdiff_t>(offset + mesh_size)),
                  component.begin());
              tensor_components[component_index] = TensorComponent{
                  component_names[component_index], std::move(component)};
            },
            tensors[component_index].data);
      }

      // Sort the tensor components by name so that they are in the same order
      // in all elements.
      alg::sort(tensor_components, [](const auto& lhs, const auto& rhs) {
        return lhs.name < rhs.name;
      });

      element_volume_data.emplace_back(
          extents[grid_index], std::move(tensor_components),
          std::vector<Spectral::Basis>(
              boost::make_transform_iterator(bases[grid_index].begin(),
                                             Spectral::to_basis),
              boost::make_transform_iterator(bases[grid_index].end(),
                                             Spectral::to_basis)),
          std::vector<Spectral::Quadrature>(
              boost::make_transform_iterator(quadratures[grid_index].begin(),
                                             Spectral::to_quadrature),
              boost::make_transform_iterator(quadratures[grid_index].end(),
                                             Spectral::to_quadrature)),
          grid_names[grid_index]);
      offset += mesh_size;
    }  // for grid_index

    // Sort the elements so they are in the same order at all time steps
    alg::sort(element_volume_data,
              [](const ElementVolumeData& lhs, const ElementVolumeData& rhs) {
                return lhs.element_name < rhs.element_name;
              });
    std::get<2>(single_time_data) = std::move(element_volume_data);
  }
  return result;
}

size_t VolumeData::get_dimension() const {
  return h5::read_value_attribute<double>(volume_data_group_.id(), "dimension");
}

std::vector<std::vector<std::string>> VolumeData::get_bases(
    const size_t observation_id) const {
  const std::string path = "ObservationId" + std::to_string(observation_id);
  detail::OpenGroup observation_group(volume_data_group_.id(), path,
                                      AccessType::ReadOnly);
  const auto dim =
      h5::read_value_attribute<size_t>(volume_data_group_.id(), "dimension");
  const auto bases_per_element = static_cast<long>(dim);

  const std::vector<int> bases_coded =
      h5::read_data<1, std::vector<int>>(observation_group.id(), "bases");
  auto all_bases = h5_detail::decode_with_dictionary_name(
      "Basis dictionary", bases_coded, observation_group);

  std::vector<std::vector<std::string>> element_bases;
  for (auto iter = all_bases.begin(); iter != all_bases.end();
       std::advance(iter, bases_per_element)) {
    element_bases.emplace_back(iter, std::next(iter, bases_per_element));
  }
  return element_bases;
}
std::vector<std::vector<std::string>> VolumeData::get_quadratures(
    const size_t observation_id) const {
  const std::string path = "ObservationId" + std::to_string(observation_id);
  detail::OpenGroup observation_group(volume_data_group_.id(), path,
                                      AccessType::ReadOnly);
  const auto dim =
      h5::read_value_attribute<size_t>(volume_data_group_.id(), "dimension");
  const auto quadratures_per_element = static_cast<long>(dim);
  const std::vector<int> quadratures_coded =
      h5::read_data<1, std::vector<int>>(observation_group.id(), "quadratures");
  auto all_quadratures = h5_detail::decode_with_dictionary_name(
      "Quadrature dictionary", quadratures_coded, observation_group);
  std::vector<std::vector<std::string>> element_quadratures;
  for (auto iter = all_quadratures.begin(); iter != all_quadratures.end();
       std::advance(iter, quadratures_per_element)) {
    element_quadratures.emplace_back(iter,
                                     std::next(iter, quadratures_per_element));
  }

  return element_quadratures;
}

}  // namespace h5

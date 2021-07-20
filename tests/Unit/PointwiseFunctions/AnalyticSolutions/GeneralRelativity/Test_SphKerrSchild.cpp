// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <string>
#include <type_traits>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/AnalyticSolutions/GeneralRelativity/VerifyGrSolution.hpp"
#include "Helpers/PointwiseFunctions/AnalyticSolutions/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/SphKerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

#include <cmath>
#include <iostream>
#include <typeinfo>

namespace {

template <typename Frame, typename DataType>
tnsr::I<DataType, 3, Frame> spatial_coords(
    const DataType& used_for_size) noexcept {
  auto x = make_with_value<tnsr::I<DataType, 3, Frame>>(used_for_size, 0.0);
  get<0>(x) = 1.32;
  get<1>(x) = 0.82;
  get<2>(x) = 1.24;
  return x;
}

// template <typename Frame, typename DataType>
// void test_schwarzschild(const DataType& used_for_size) noexcept {

//   // Parameters for SphKerrSchild solution
//   const double mass = 1.01;
//   const std::array<double, 3> spin{{0.0, 0.0, 0.0}};
//   const std::array<double, 3> center{{0.0, 0.0, 0.0}};
//   const auto x = spatial_coords<Frame>(used_for_size);
//   const double t = 1.3;

//   std::cout << "this is the value of x:" << x << "\n";

//   // Evaluate solution
//   gr::Solutions::SphKerrSchild solution(mass, spin, center);

template <typename Frame, typename DataType>
void test_sph_kerr_schild(const DataType& used_for_size) noexcept {
  // Parameters for SphKerrSchild solution
  const double mass = 1.01;
  const std::array<double, 3> spin{{0.0, 0.0, 0.0}};
  const std::array<double, 3> center{{0.0, 0.0, 0.0}};
  const auto x = spatial_coords<Frame>(used_for_size);
  //   std::cout << x << "\n";
  // const double t = 1.3;

  // Evaluate solution, instantiating a spherical kerrschild and calling
  // constructor
  gr::Solutions::SphKerrSchild solution(mass, spin, center);

  //     const auto vars = solution.variables(
  //       x, t, typename gr::Solutions::SphKerrSchild::tags<DataType,
  //       Frame>{});
  //   const auto& lapse = get<gr::Tags::Lapse<DataType>>(vars);
  //   const auto& dt_lapse = get<Tags::dt<gr::Tags::Lapse<DataType>>>(vars);
  //   const auto& d_lapse =
  //       get<typename gr::Solutions::KerrSchild::DerivLapse<DataType, Frame>>(
  //           vars);
  //   const auto& shift = get<gr::Tags::Shift<3, Frame, DataType>>(vars);
  //   const auto& d_shift =
  //       get<typename gr::Solutions::KerrSchild::DerivShift<DataType, Frame>>(
  //           vars);
  //   const auto& dt_shift =
  //       get<Tags::dt<gr::Tags::Shift<3, Frame, DataType>>>(vars);
  //   const auto& g = get<gr::Tags::SpatialMetric<3, Frame, DataType>>(vars);
  //   const auto& dt_g =
  //       get<Tags::dt<gr::Tags::SpatialMetric<3, Frame, DataType>>>(vars);
  //   const auto& d_g = get<
  //       typename gr::Solutions::KerrSchild::DerivSpatialMetric<DataType,
  //       Frame>>( vars);
  // }
}

// Set up the x coords or spatial coords ??
const tnsr::i<DataVector, 3, Frame::Inertial> give_values() noexcept {
  tnsr::i<DataVector, 3, Frame::Inertial> empty{3, 0.0};

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      if (i == j) {
        empty.get(i)[j] = 2.0;
      }
    }
  }
  return empty;
}

const DataVector r_squared(3, 4.0);

const DataVector r(3, sqrt(r_squared[0]));

// const std::array<double, 3> spin{{0.0, 0.0, 3.0}};

const DataVector make_spin_a() {
  DataVector empty(3, 0.0);

  //   // auto empty = make_with_value<tnsr::i<DataType, 3, Frame::Inertial>>(3,
  //   0.0);
  //   // // tnsr::i<DataVector, 3, Frame::Inertial> empty(3,0.0);
  //   // for (size_t i = 0; i < 3; ++i) {
  //   //   for (size_t j = 0; j < 3; ++j) {
  //   //     empty.get(i,j) =
  //   //   }
  //   // }

  empty[0] = 0.0;
  empty[1] = 0.0;
  empty[2] = 3.0;
  return empty;
}

const DataVector square_a(const DataVector spin_a) {
  DataVector empty(3, 0.0);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      empty[i] += spin_a[j] * spin_a[j];
    }
  }
  return empty;
}

// auto make_a_squared(const std::array<double, 3> spin_a){
//    auto a_squared = std::inner_product(spin_a.begin(), spin_a.end(),
//    spin_a.begin(), 0.); return a_squared;
// }

}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticSolutions.Gr.SphKerrSchild",
                  "[PointwiseFunctions][Unit]") {
  auto x_coords = give_values();

  std::cout << "this is x coords:" << x_coords << "\n";
  std::cout << "this is r squared:" << r_squared << "\n";
  std::cout << "this is r:" << r << "\n";

  auto spin_a = make_spin_a();

  auto a_squared = square_a(spin_a);

  std::cout << "this is spin_a:" << spin_a << "\n";
  // std::cout << "this is a_squared:" << make_a_squared(spin) << "\n";
  std::cout << "this is a_squared:" << a_squared << "\n";
}

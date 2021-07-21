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

// Set up x_coords
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

// Set up r_squared
const DataVector r_squared(3, 4.0);
// Set up r
const DataVector r(3, sqrt(r_squared[0]));

// Set up spin_a
const DataVector make_spin_a() {
  DataVector empty(3, 0.0);

  empty[0] = 0.0;
  empty[1] = 0.0;
  empty[2] = 3.0;

  return empty;
}

// Set up a_squared
const DataVector make_a_squared(const DataVector spin_a) {
  DataVector empty(3, 0.0);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      empty[i] += spin_a[j] * spin_a[j];
    }
  }
  return empty;
}

// Set up rho
template <typename DataType>
const DataVector make_rho(const DataType r_squared,
                          const DataType a_squared) noexcept {
  DataVector empty(3, 0.0);

  for (size_t i = 0; i < 3; ++i) {
    empty[i] = sqrt(r_squared[i] + a_squared[i]);
  }

  return empty;
}

// Set up a_dot_x
template <typename DataType>
const DataVector make_a_dot_x(
    const tnsr::i<DataType, 3, Frame::Inertial> x_coords,
    const DataType spin_a) noexcept {
  DataVector empty(3, 0.0);

  for (size_t i = 0; i < 3; ++i) {
    auto x_coords_vector = x_coords[i];
    empty[i] = std::inner_product(spin_a.begin(), spin_a.end(),
                                  x_coords_vector.begin(), 0.);
  }

  return empty;
}

// Set up matrix_F
template <typename DataType>
const tnsr::Ij<DataVector, 3, Frame::Inertial> make_matrix_F(
    const DataType r, const DataType rho, const DataType spin_a,
    const DataType a_squared) {
  tnsr::Ij<DataVector, 3, Frame::Inertial> matrix_F{3, 0.0};

  // Setup
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      matrix_F.get(i, j) = -1. / rho / cube(r);
    }
  }

  // F Matrix
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      if (i == j) {
        matrix_F.get(i, j) *= (a_squared[i] - spin_a[i] * spin_a[j]);
      } else {
        matrix_F.get(i, j) *= -spin_a[i] * spin_a[j];
      }
    }
  }

  return matrix_F;
}

// Test matrix_F
template <typename DataType>
const tnsr::Ij<DataVector, 3, Frame::Inertial> test_matrix_F(
    const tnsr::Ij<DataVector, 3, Frame::Inertial> matrix_F, const DataType r,
    const DataType rho, const DataType a_squared) {
  tnsr::Ij<DataVector, 3, Frame::Inertial> test_matrix_F{3, 0.0};

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      test_matrix_F.get(i, j) =
          matrix_F.get(i, j) * (-(rho * cube(r)) / a_squared);
    }
  }

  return test_matrix_F;
}

// Set up matrix_P
template <typename DataType>
const tnsr::Ij<DataVector, 3, Frame::Inertial> make_matrix_P(
    const DataType r, const DataType rho, const DataType spin_a) noexcept {
  tnsr::Ij<DataVector, 3, Frame::Inertial> matrix_P{3, 0.0};

  // Set up
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      matrix_P.get(i, j) = -1. / (rho + r) / r;
    }
  }

  // P matrix
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      if (i == j) {
        matrix_P.get(i, j) *= spin_a[i] * spin_a[j];
        matrix_P.get(i, j) += rho / r;
      } else {
        matrix_P.get(i, j) *= spin_a[i] * spin_a[j];
      }
    }
  }

  return matrix_P;
}

// Test matrix_P
template <typename DataType>
const tnsr::Ij<DataVector, 3, Frame::Inertial> test_matrix_P(
    const tnsr::Ij<DataVector, 3, Frame::Inertial> matrix_P, const DataType r,
    const DataType rho) {
  tnsr::Ij<DataVector, 3, Frame::Inertial> test_matrix_P{3, 0.0};

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      if (i == j) {
        test_matrix_P.get(i, j) = matrix_P.get(i, j) * (r / rho);
      }
    }
  }

  return test_matrix_P;
}

// Set up jacobian
template <typename DataType>
const tnsr::Ij<DataType, 3, Frame::Inertial> make_jacobian(
    const tnsr::i<DataType, 3, Frame::Inertial> x_coords,
    const tnsr::Ij<DataType, 3, Frame::Inertial> matrix_F,
    const tnsr::Ij<DataType, 3, Frame::Inertial> matrix_P) {
  tnsr::Ij<DataType, 3, Frame::Inertial> jacobian(3, 0.0);

  // Jacobian
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      jacobian.get(i, j) = matrix_P.get(i, j);
      for (size_t k = 0; k < 3; ++k) {
        jacobian.get(i, j) +=
            matrix_F.get(i, k) * x_coords.get(k) * x_coords.get(j);
      }
    }
  }

  return jacobian;
}

}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticSolutions.Gr.SphKerrSchild",
                  "[PointwiseFunctions][Unit]") {
  auto x_coords = give_values();

  std::cout << "This is x coords:"
            << "\n"
            << x_coords << "\n";
  std::cout << "This is r squared:"
            << "\n"
            << r_squared << "\n";
  std::cout << "This is r:"
            << "\n"
            << r << "\n";

  auto spin_a = make_spin_a();
  std::cout << "This is spin_a:"
            << "\n"
            << spin_a << "\n";

  auto a_squared = make_a_squared(spin_a);
  std::cout << "This is a_squared:"
            << "\n"
            << a_squared << "\n";

  auto rho = make_rho(r_squared, a_squared);
  std::cout << "This is rho:"
            << "\n"
            << rho << "\n";

  auto a_dot_x = make_a_dot_x(x_coords, spin_a);
  std::cout << "This is a_dot_x:"
            << "\n"
            << a_dot_x << "\n";

  auto matrix_F = make_matrix_F(r, rho, spin_a, a_squared);
  std::cout << "This is Matrix F:"
            << "\n"
            << matrix_F << "\n";
  auto tested_matrix_F = test_matrix_F(matrix_F, r, rho, a_squared);
  std::cout << "This is the tested Matrix F:"
            << "\n"
            << tested_matrix_F << "\n";

  auto matrix_P = make_matrix_P(r, rho, spin_a);
  std::cout << "This is Matrix P:"
            << "\n"
            << matrix_P << "\n";
  auto tested_matrix_P = test_matrix_P(matrix_P, r, rho);
  std::cout << "This is the tested Matrix P:"
            << "\n"
            << tested_matrix_P << "\n";

  auto jacobian = make_jacobian(x_coords, matrix_F, matrix_P);
  std::cout << "This is the Jacobian:"
            << "\n"
            << jacobian << "\n";
}

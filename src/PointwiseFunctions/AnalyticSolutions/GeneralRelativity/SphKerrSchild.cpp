// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/SphKerrSchild.hpp"

#include <cmath>  // IWYU pragma: keep
#include <numeric>
#include <ostream>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/GeneralRelativity/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/StdHelpers.hpp"

#include <iostream>

namespace gr::Solutions {

SphKerrSchild::SphKerrSchild(const double mass,
                             SphKerrSchild::Spin::type dimensionless_spin,
                             SphKerrSchild::Center::type center,
                             const Options::Context& context)
    : mass_(mass),
      // clang-tidy: do not std::move trivial types.
      dimensionless_spin_(std::move(dimensionless_spin)),  // NOLINT
      // clang-tidy: do not std::move trivial types.
      center_(std::move(center))  // NOLINT
{
  const double spin_magnitude = magnitude(dimensionless_spin_);
  if (spin_magnitude > 1.0) {
    PARSE_ERROR(context, "Spin magnitude must be < 1. Given spin: "
                             << dimensionless_spin_ << " with magnitude "
                             << spin_magnitude);
  }
  if (mass_ < 0.0) {
    PARSE_ERROR(context, "Mass must be non-negative. Given mass: " << mass_);
  }
}

void SphKerrSchild::pup(PUP::er& p) noexcept {
  p | mass_;
  p | dimensionless_spin_;
  p | center_;
}

// what is this?
template <typename DataType, typename Frame>
SphKerrSchild::IntermediateComputer<DataType, Frame>::IntermediateComputer(
    const SphKerrSchild& solution, const tnsr::I<DataType, 3, Frame>& x,
    const double null_vector_0) noexcept
    : solution_(solution), x_(x), null_vector_0_(null_vector_0) {}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3, Frame>*> x_sph_minus_center,
    const gsl::not_null<CachedBuffer*> /*cache*/,
    internal_tags::x_sph_minus_center<DataType, Frame> /*meta*/)
    const noexcept {
  *x_sph_minus_center = x_;  // why are you here?
  for (size_t d = 0; d < 3; ++d) {
    x_sph_minus_center->get(d) -= gsl::at(solution_.center(), d);
  }
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> r_squared,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::r_squared<DataType> /*meta*/) const noexcept {
  const auto& x_sph_minus_center =
      cache->get_var(internal_tags::x_sph_minus_center<DataType, Frame>{});

  for (size_t i = 0; i < 3; ++i) {
    r_squared->get(i) = r_squared += square(x_sph_minus_center.get(i))
  }
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> r,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::r<DataType> /*meta*/) const noexcept {
  const auto& r_squared =
      get(cache->get_var(internal_tags::r_squared<DataType>{}));

  get(*r) = sqrt(r_squared);
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> rho,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::rho<DataType> /*meta*/) const noexcept {
  const auto spin_a = solution_.dimensionless_spin() * solution_.mass();
  const auto& r_squared =
      get(cache->get_var(internal_tags::r_squared<DataType>{}));
  const auto a_squared =
      std::inner_product(spin_a.begin(), spin_a.end(), spin_a.begin(), 0.);
  get(*rho) = sqrt(r_squared + a_squared);
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> a_dot_x,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::a_dot_x<DataType> /*meta*/) const noexcept {
  const auto& x_sph_minus_center =
      cache->get_var(internal_tags::x_sph_minus_center<DataType, Frame>{});

  const auto spin_a = solution_.dimensionless_spin() * solution_.mass();
  get(*a_dot_x) = spin_a[0] * get<0>(x_sph_minus_center) +
                  spin_a[1] * get<1>(x_sph_minus_center) +
                  spin_a[2] * get<2>(x_sph_minus_center);
}

// incomplete (TEST)
template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::Ik<DataType, 3, Frame>*> F_matrix,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::F_matrix<DataType> /*meta*/) const noexcept {
  const auto spin_a = solution_.dimensionless_spin() * solution_.mass();
  const auto a_squared =
      std::inner_product(spin_a.begin(), spin_a.end(), spin_a.begin(), 0.);
  const auto& rho = get(cache->get_var(internal_tags::rho<DataType>{}));
  const auto& r_squared =
      get(cache->get_var(internal_tags::r_squared<DataType>{}));

  auto F_matrix = tnsr::ij<DataVector, Dim, Frame::Inertial> matrix_F{
      9, -1. / rho / cube(sqrt(r_squared))};

  // std::cout << "the value of matrix_F is" << matrix_F << "\n";

  // F matrix
  for (int i = 0; i < 3; ++i) {
    for (int j = i; j < 3; ++j) {
      if (i == j) {
        F_matrix.get(i, j) *= (a_squared - spin_a[i] * spin_a[j]);
      } else {
        F_matrix.get(i, j) *= -spin_a[i] * spin_a[j];
      }
    }
  }
}

// incomplete (TEST)
template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> P_matrix,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::P_matrix<DataType> /*meta*/) const noexcept {
  const auto spin_a = solution_.dimensionless_spin() * solution_.mass();
  const auto a_squared =
      std::inner_product(spin_a.begin(), spin_a.end(), spin_a.begin(), 0.);
  const auto& rho = get(cache->get_var(internal_tags::rho<DataType>{}));

  auto P_matrix = tnsr::Ij<DataVector, Dim, Frame::Inertial> P_matrix{
      9, -1 / (rho + r) / r};

  // P matrix
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      if (i == j) {
        P_matrix.get(i, j) *= spin_a[i] * spin_a[j];
        P_matrix.get(i, j) += rho / r;
      } else {
        P_matrix.get(i, j) *= spin_a[i] * spin_a[j];
      }
    }
  }
}

// incomplete (TEST)
template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> jacobian const
        gsl::not_null<CachedBuffer*>
            cache,
    internal_tags::jacobian<DataType> /*meta*/) const noexcept {
  auto jacobian =
      tnsr::Ij<DataVector, Dim, Frame::Inertial> jacobian{9, P_matrix};

  // Jacobian
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        jacobian.get(i, j) +=
            F_matrix.get(i, k) * x_sph_minus_center[k] * x_sph_minus_center[j];
      }
    }
  }
}

// incomplete (TEST)
template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> D_matrix,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::D_matrix<DataType> /*meta*/) const noexcept {
  const auto spin_a = solution_.dimensionless_spin() * solution_.mass();
  const auto a_squared =
      std::inner_product(spin_a.begin(), spin_a.end(), spin_a.begin(), 0.);
  const auto& rho = get(cache->get_var(internal_tags::rho<DataType>{}));
  const auto& r_squared =
      get(cache->get_var(internal_tags::r_squared<DataType>{}));

  auto D_matrix = tnsr::Ij<DataVector, Dim, Frame::Inertial> D_matrix{
      9, 1. / cube(rho) / sqrt(r_squared)};

  // D matrix
  for (int i = 0; i < 3; ++i) {
    for (int j = i; j < 3; ++j) {
      if (i == j) {
        D_matrix.get(i, j) *= (a_squared - spin_a[i] * spin_a[j]);
      } else {
        D_matrix.get(i, j) *= -spin_a[i] * spin_a[j];
      }
    }
  }
}

// incomplete (TEST)
template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> C_matrix,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::C_matrix<DataType> /*meta*/) const noexcept {
  auto C_matrix =
      tnsr::Ij<DataVector, Dim, Frame::Inertial> C_matrix{9, D_matrix};

  // D matrix
  for (int i = 0; i < 3; ++i) {
    for (int j = i; j < 3; ++j) {
      C_matrix.get(i, j) += -3. * F_matrix.get(i, j);
    }
  }
}

// incomplete (TEST)
template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> deriv_jacobian const
        gsl::not_null<CachedBuffer*>
            cache,
    internal_tags::deriv_jacobian<DataType> /*meta*/) const noexcept {
  const auto& x_sph_minus_center =
      cache->get_var(internal_tags::x_sph_minus_center<DataType, Frame>{});
  const auto& r_squared =
      get(cache->get_var(internal_tags::r_squared<DataType>{}));

  auto deriv_jacobian =
      tnsr::Ij<DataVector, Dim, Frame::Inertial> deriv_jacobian{9, F_matrix};

  // deriv_Jacobian
  for (int k = 0; k < 3; ++k) {
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        deriv_jacobian.get(i, j, k) =
            F_matrix.get(i, j) * x_sph_minus_center[k] +
            F_matrix.get(i, k) * x_sph_minus_center[j];
        for (int m = 0; m < 3; ++m) {
          if (j == k) {  // j==k acts as a Kronecker delta
            deriv_jacobian.get(i, j, k) +=
                F_matrix.get(i, m) * x_sph_minus_center[m];
          }
          deriv_jacobian.get(i, j, k) +=
              C_matrix.get(i, m) * x_sph_minus_center[k] *
              x_sph_minus_center[m] * x_sph_minus_center[j] / r_squared;
        }
      }
    }
  }
}

}  // namespace gr::Solutions

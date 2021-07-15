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
  *x_sph_minus_center = x_;
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
    r_squared->get(i) += square(x_sph_minus_center.get(i))
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

// TEST
template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> matrix_F,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::matrix_F<DataType> /*meta*/) const noexcept {
  const auto spin_a = solution_.dimensionless_spin() * solution_.mass();
  const auto a_squared =
      std::inner_product(spin_a.begin(), spin_a.end(), spin_a.begin(), 0.);
  const auto& rho = get(cache->get_var(internal_tags::rho<DataType>{}));
  const auto& r = get(cache->get_var(internal_tags::r<DataType>{}));

  auto matrix_F = tnsr::ij<DataVector, Dim, Frame::Inertial> matrix_F{
      9, -1. / rho / cube(r)};

  // F matrix
  for (int i = 0; i < 3; ++i) {
    for (int j = i; j < 3; ++j) {
      if (i == j) {
        matrix_F->get(i, j) *= (a_squared - spin_a.get(i) * spin_a.get(j));
      } else {
        matrix_F->get(i, j) *= -spin_a.get(i) * spin_a.get(j);
      }
    }
  }
}

// TEST
template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> matrix_P,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::matrix_P<DataType> /*meta*/) const noexcept {
  const auto spin_a = solution_.dimensionless_spin() * solution_.mass();
  const auto a_squared =
      std::inner_product(spin_a.begin(), spin_a.end(), spin_a.begin(), 0.);
  const auto& rho = get(cache->get_var(internal_tags::rho<DataType>{}));

  auto matrix_P = tnsr::Ij<DataVector, Dim, Frame::Inertial> matrix_P{
      9, -1. / (rho + r) / r};

  // P matrix
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      if (i == j) {
        matrix_P->get(i, j) *= spin_a.get(i) * spin_a.get(j);
        matrix_P->get(i, j) += rho / r;
      } else {
        matrix_P->get(i, j) *= spin_a.get(i) * spin_a.get(j);
      }
    }
  }
}

// TEST
template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> jacobian const
        gsl::not_null<CachedBuffer*>
            cache,
    internal_tags::jacobian<DataType> /*meta*/) const noexcept {
  const auto& x_sph_minus_center =
      cache->get_var(internal_tags::x_sph_minus_center<DataType, Frame>{});
  const auto& matrix_F =
      cache->get_var(internal_tags::matrix_F<DataType, Frame>{});
  const auto& matrix_P =
      cache->get_var(internal_tags::matrix_P<DataType, Frame>{});
  auto jacobian = tnsr::Ij<DataVector, Dim, Frame::Inertial> jacobian{9, 0.};

  // Jacobian
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      jacobian->get(i, j) = matrix_P.get(i, j) for (int k = 0; k < 3; ++k) {
        jacobian->get(i, j) += matrix_F.get(i, k) * x_sph_minus_center.get(k) *
                               x_sph_minus_center.get(j);
      }
    }
  }
}

// TEST
template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> matrix_D,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::matrix_D<DataType> /*meta*/) const noexcept {
  const auto spin_a = solution_.dimensionless_spin() * solution_.mass();
  const auto a_squared =
      std::inner_product(spin_a.begin(), spin_a.end(), spin_a.begin(), 0.);
  const auto& rho = get(cache->get_var(internal_tags::rho<DataType>{}));
  const auto& r = get(cache->get_var(internal_tags::r<DataType>{}));

  auto matrix_D = tnsr::Ij<DataVector, Dim, Frame::Inertial> matrix_D{
      9, 1. / cube(rho) / r};

  // D matrix
  for (int i = 0; i < 3; ++i) {
    for (int j = i; j < 3; ++j) {
      if (i == j) {
        matrix_D->get(i, j) *= (a_squared - spin_a.get(i) * spin_a.get(j));
      } else {
        matrix_D->get(i, j) *= -spin_a.get(i) * spin_a.get(j);
      }
    }
  }
}

// TEST
template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> matrix_C,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::matrix_C<DataType> /*meta*/) const noexcept {
  const auto& matrix_F =
      cache->get_var(internal_tags::matrix_F<DataType, Frame>{});
  const auto& matrix_D =
      cache->get_var(internal_tags::matrix_D<DataType, Frame>{});

  auto matrix_C = tnsr::Ij<DataVector, Dim, Frame::Inertial> C_matrix{9, 0.};

  // C matrix
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      matrix_C->get(i, j) = matrix_D.get(i, j) - 3. * matrix_F.get(i, j);
    }
  }
}

// TEST
template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> deriv_jacobian const
        gsl::not_null<CachedBuffer*>
            cache,
    internal_tags::deriv_jacobian<DataType> /*meta*/) const noexcept {
  const auto& matrix_C =
      cache->get_var(internal_tags::matrix_C<DataType, Frame>{});
  const auto& matrix_F =
      cache->get_var(internal_tags::matrix_F<DataType, Frame>{});
  const auto& x_sph_minus_center =
      cache->get_var(internal_tags::x_sph_minus_center<DataType, Frame>{});
  const auto& r_squared =
      get(cache->get_var(internal_tags::r_squared<DataType>{}));

  auto deriv_jacobian =
      tnsr::Ij<DataVector, Dim, Frame::Inertial> deriv_jacobian{9, 0.};

  // deriv_Jacobian
  for (int k = 0; k < 3; ++k) {
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        deriv_jacobian->get(i, j, k) =
            matrix_F.get(i, j) * x_sph_minus_center.get(k) +
            matrix_F.get(i, k) * x_sph_minus_center.get(j);
        for (int m = 0; m < 3; ++m) {
          if (j == k) {  // j==k acts as a Kronecker delta
            deriv_jacobian->get(i, j, k) +=
                matrix_F.get(i, m) * x_sph_minus_center.get(m);
          }
          deriv_jacobian->get(i, j, k) +=
              matrix_C.get(i, m) * x_sph_minus_center.get(k) *
              x_sph_minus_center.get(m) * x_sph_minus_center.get(j) / r_squared;
        }
      }
    }
  }
}

// TEST
template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> matrix_Q const
        gsl::not_null<CachedBuffer*>
            cache,
    internal_tags::matrix_Q<DataType> /*meta*/) const noexcept {
  const auto spin_a = solution_.dimensionless_spin() * solution_.mass();
  const auto& rho = get(cache->get_var(internal_tags::rho<DataType>{}));
  const auto& r = get(cache->get_var(internal_tags::r<DataType>{}));

  auto matrix_Q = tnsr::Ij<DataVector, Dim, Frame::Inertial> matrix_Q{
      9, 1. / (rho + r) / rho};

  // Q matrix
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      if (i == j) {
        matrix_Q->get(i, j) *= spin_a.get(i) * spin_a.get(j);
        matrix_Q->get(i, j) += r / rho;
      } else {
        matrix_Q->get(i, j) *= spin_a.get(i) * spin_a.get(j);
      }
    }
  }
}

// TEST
template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> matrix_G1,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::matrix_G1<DataType> /*meta*/) const noexcept {
  const auto spin_a = solution_.dimensionless_spin() * solution_.mass();
  const auto a_squared =
      std::inner_product(spin_a.begin(), spin_a.end(), spin_a.begin(), 0.);
  const auto& rho = get(cache->get_var(internal_tags::rho<DataType>{}));
  const auto& r = get(cache->get_var(internal_tags::r<DataType>{}));

  auto matrix_G1 = tnsr::ij<DataVector, Dim, Frame::Inertial> matrix_G1{
      9, 1. / sqr(rho) / r};

  for (int i = 0; i < 3; ++i) {
    for (int j = i; j < 3; ++j) {
      if (i == j) {
        matrix_G1->get(i, j) *= (a_squared - spin_a.get(i) * spin_a.get(j));
      } else {
        matrix_G1->get(i, j) *= -spin_a.get(i) * spin_a.get(j);
      }
    }
  }
}

// ADD TO HPP
template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> s_number,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::s_number<DataType> /*meta*/) const noexcept {
  const auto& r_squared =
      get(cache->get_var(internal_tags::r_squared<DataType>{}));
  const auto& a_dot_x = get(cache->get_var(internal_tags::a_dot_x<DataType>{}));

  get(*s_number) = r_squared + sqr(a_dot_x) / r_squared;
}
}  // namespace gr::Solutions

// TEST
template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> matrix_G2,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::matrix_G2<DataType> /*meta*/) const noexcept {
  const auto& matrix_Q =
      cache->get_var(internal_tags::matrix_Q<DataType, Frame>{});
  const auto& rho = get(cache->get_var(internal_tags::rho<DataType>{}));
  const auto& r = get(cache->get_var(internal_tags::r<DataType>{}));
  const auto& r_squared =
      get(cache->get_var(internal_tags::r_squared<DataType>{}));
  const auto& s_number =
      get(cache->get_var(internal_tags::s_number<DataType>{}));

  auto matrix_G2 =
      tnsr::ij<DataVector, Dim, Frame::Inertial> matrix_G2{9, sqr(rho) / r};

  for (int i = 0; i < 3; ++i) {
    for (int j = i; j < 3; ++j) {
      matrix_G2->get(i, j) *= matrix_Q.get(i, j) / s_number
    }
  }
}

// TEST
template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> inv_jacobian const
        gsl::not_null<CachedBuffer*>
            cache,
    internal_tags::inv_jacobian<DataType> /*meta*/) const noexcept {
  const auto& matrix_Q =
      cache->get_var(internal_tags::matrix_Q<DataType, Frame>{});
  const auto& matrix_G1 =
      cache->get_var(internal_tags::matrix_G1<DataType, Frame>{});
  const auto& matrix_G2 =
      cache->get_var(internal_tags::matrix_G2<DataType, Frame>{});
  const auto& x_sph_minus_center =
      cache->get_var(internal_tags::x_sph_minus_center<DataType, Frame>{});

  auto inv_jacobian =
      tnsr::Ij<DataVector, Dim, Frame::Inertial> inv_jacobian{9, 0.};

  // G1dotx^i: G1^i_m xhat^m
  // G2dotx_j: G2^n_j xhat_n
  for (int i = 0; i < 3; ++i) {
    for (int m = 0; m < 3; ++m) {
      const auto G1_dot_x[i] += matrix_G1.get(i, m) * x_sph_minus_center.get(m);
      const auto G2_dot_x[i] += matrix_G2.get(m, i) * x_sph_minus_center.get(m);
    }
  }
  // InvJacobian
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      inv_jacobian->get(i, j) =
          matrix_Q.get(i, j) + G1_dot_x.get(i) * G2_dot_x.get(j);
    }
  }
}

// missing matrix_E1, matrix_E2, deriv_inv_jacobian ???

// TEST
template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3, Frame>*> x_kerr_schild,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::x_kerr_schild<DataType> /*meta*/) const noexcept {
  const auto spin_a = solution_.dimensionless_spin() * solution_.mass();
  const auto& x_sph_minus_center =
      cache->get_var(internal_tags::x_sph_minus_center<DataType, Frame>{});
  const auto& a_dot_x = get(cache->get_var(internal_tags::a_dot_x<DataType>{}));
  const auto& r = get(cache->get_var(internal_tags::r<DataType>{}));
  const auto& rho = get(cache->get_var(internal_tags::rho<DataType>{}));

  for (int i = 0; i < 3; ++i) {
    x_kerr_schild->get(i) = rho / r * x_sph_minus_center.get(i) -
                            spin_a.get(i) * a_dot_x / r / (rho + r);
  }
}

// template <typename DataType, typename Frame>
// SphKerrSchild::IntermediateVars<DataType, Frame>::IntermediateVars(
//     const SphKerrSchild& solution, const tnsr::I<DataType, 3, Frame>& x)
//     noexcept : CachedBuffer(get_size(::get<0>(x)),
//     IntermediateComputer<DataType, Frame>(
//                                               solution, x, null_vector_0_))
//                                               {}

}  // namespace gr::Solutions

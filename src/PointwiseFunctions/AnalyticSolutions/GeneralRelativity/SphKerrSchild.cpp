// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/SphKerrSchild.hpp"

#include <cmath>  // IWYU pragma: keep
#include <numeric>
#include <ostream>
#include <typeinfo>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/EagerMath/CrossProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
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

void SphKerrSchild::pup(PUP::er& p) {
  p | mass_;
  p | dimensionless_spin_;
  p | center_;
}

template <typename DataType, typename Frame>
SphKerrSchild::IntermediateComputer<DataType, Frame>::IntermediateComputer(
    const SphKerrSchild& solution, const tnsr::I<DataType, 3, Frame>& x)
    : solution_(solution), x_(x) {}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3, Frame>*> x_sph_minus_center,
    const gsl::not_null<CachedBuffer*> /*cache*/,
    internal_tags::x_sph_minus_center<DataType, Frame> /*meta*/) const {
  *x_sph_minus_center = x_;

  for (size_t i = 0; i < 3; ++i) {
    x_sph_minus_center->get(i) -= gsl::at(solution_.center(), i);
  }
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> r_squared,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::r_squared<DataType> /*meta*/) const {
  const auto& x_sph_minus_center = cache->get_var(
      *this, internal_tags::x_sph_minus_center<DataType, Frame>{});

  r_squared->get() = square(x_sph_minus_center.get(0));
  for (size_t i = 1; i < 3; ++i) {
    r_squared->get() += square(x_sph_minus_center.get(i));
  }
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> r,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::r<DataType> /*meta*/) const {
  const auto& r_squared =
      get(cache->get_var(*this, internal_tags::r_squared<DataType>{}));

  get(*r) = sqrt(r_squared);
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> rho,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::rho<DataType> /*meta*/) const {
  const auto spin_a = solution_.dimensionless_spin();
  const auto& r_squared =
      get(cache->get_var(*this, internal_tags::r_squared<DataType>{}));
  const auto a_squared =
      std::inner_product(spin_a.begin(), spin_a.end(), spin_a.begin(), 0.);

  get(*rho) = sqrt(r_squared + a_squared);
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> matrix_F,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::matrix_F<DataType, Frame> /*meta*/) const {
  const auto spin_a = solution_.dimensionless_spin();
  const auto a_squared =
      std::inner_product(spin_a.begin(), spin_a.end(), spin_a.begin(), 0.);
  const auto& rho = get(cache->get_var(*this, internal_tags::rho<DataType>{}));
  const auto& r = get(cache->get_var(*this, internal_tags::r<DataType>{}));

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      matrix_F->get(i, j) = -1. / rho / cube(r);
    }
  }
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      // Kronecker delta
      if (i == j) {
        matrix_F->get(i, j) *=
            (a_squared - gsl::at(spin_a, i) * gsl::at(spin_a, j));
      } else {
        matrix_F->get(i, j) *= -gsl::at(spin_a, i) * gsl::at(spin_a, j);
      }
    }
  }
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> matrix_P,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::matrix_P<DataType, Frame> /*meta*/) const {
  const auto spin_a = solution_.dimensionless_spin();
  const auto& rho = get(cache->get_var(*this, internal_tags::rho<DataType>{}));
  const auto& r = get(cache->get_var(*this, internal_tags::r<DataType>{}));

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      matrix_P->get(i, j) = -1. / (rho + r) / r;
    }
  }
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      if (i == j) {
        matrix_P->get(i, j) *= gsl::at(spin_a, i) * gsl::at(spin_a, j);
        matrix_P->get(i, j) += rho / r;
      } else {
        matrix_P->get(i, j) *= gsl::at(spin_a, i) * gsl::at(spin_a, j);
      }
    }
  }
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> jacobian,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::jacobian<DataType, Frame> /*meta*/) const {
  const auto& x_sph_minus_center = cache->get_var(
      *this, internal_tags::x_sph_minus_center<DataType, Frame>{});
  const auto& matrix_P =
      cache->get_var(*this, internal_tags::matrix_P<DataType, Frame>{});
  const auto& matrix_F =
      cache->get_var(*this, internal_tags::matrix_F<DataType, Frame>{});

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      jacobian->get(j, i) = matrix_P.get(i, j);
      for (size_t k = 0; k < 3; ++k) {
        jacobian->get(j, i) += matrix_F.get(i, k) * x_sph_minus_center.get(k) *
                               x_sph_minus_center.get(j);
      }
    }
  }
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> matrix_D,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::matrix_D<DataType, Frame> /*meta*/) const {
  const auto spin_a = solution_.dimensionless_spin();
  const auto a_squared =
      std::inner_product(spin_a.begin(), spin_a.end(), spin_a.begin(), 0.);
  const auto& rho = get(cache->get_var(*this, internal_tags::rho<DataType>{}));
  const auto& r = get(cache->get_var(*this, internal_tags::r<DataType>{}));

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      matrix_D->get(i, j) = 1. / cube(rho) / r;
    }
  }
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      // Kronecker delta
      if (i == j) {
        matrix_D->get(i, j) *=
            (a_squared - gsl::at(spin_a, i) * gsl::at(spin_a, j));
      } else {
        matrix_D->get(i, j) *= -gsl::at(spin_a, i) * gsl::at(spin_a, j);
      }
    }
  }
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> matrix_C,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::matrix_C<DataType, Frame> /*meta*/) const {
  const auto& matrix_F =
      cache->get_var(*this, internal_tags::matrix_F<DataType, Frame>{});
  const auto& matrix_D =
      cache->get_var(*this, internal_tags::matrix_D<DataType, Frame>{});

  for (size_t i = 0; i < 3; ++i) {
    for (size_t m = 0; m < 3; ++m) {
      matrix_C->get(i, m) = matrix_D.get(i, m) - 3. * matrix_F.get(i, m);
    }
  }
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::iJk<DataType, 3, Frame>*> deriv_jacobian,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::deriv_jacobian<DataType, Frame> /*meta*/) const {
  const auto& matrix_C =
      cache->get_var(*this, internal_tags::matrix_C<DataType, Frame>{});
  const auto& matrix_F =
      cache->get_var(*this, internal_tags::matrix_F<DataType, Frame>{});
  const auto& x_sph_minus_center = cache->get_var(
      *this, internal_tags::x_sph_minus_center<DataType, Frame>{});
  const auto& r_squared =
      get(cache->get_var(*this, internal_tags::r_squared<DataType>{}));

  for (size_t k = 0; k < 3; ++k) {
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        deriv_jacobian->get(k, j, i) =
            matrix_F.get(i, j) * x_sph_minus_center.get(k) +
            matrix_F.get(i, k) * x_sph_minus_center.get(j);

        for (size_t m = 0; m < 3; ++m) {
          // Kronecker delta
          if (j == k) {
            deriv_jacobian->get(k, j, i) +=
                matrix_F.get(i, m) * x_sph_minus_center.get(m);
          }
          deriv_jacobian->get(k, j, i) +=
              matrix_C.get(i, m) * x_sph_minus_center.get(k) *
              x_sph_minus_center.get(m) * x_sph_minus_center.get(j) / r_squared;
        }
      }
    }
  }
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> matrix_Q,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::matrix_Q<DataType, Frame> /*meta*/) const {
  const auto spin_a = solution_.dimensionless_spin();
  const auto& rho = get(cache->get_var(*this, internal_tags::rho<DataType>{}));
  const auto& r = get(cache->get_var(*this, internal_tags::r<DataType>{}));

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      matrix_Q->get(i, j) = 1. / ((rho + r) * rho);
    }
  }
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      // Kronecker delta
      if (i == j) {
        matrix_Q->get(i, j) *= gsl::at(spin_a, i) * gsl::at(spin_a, j);
        matrix_Q->get(i, j) += r / rho;
      } else {
        matrix_Q->get(i, j) *= gsl::at(spin_a, i) * gsl::at(spin_a, j);
      }
    }
  }
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> matrix_G1,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::matrix_G1<DataType, Frame> /*meta*/) const {
  const auto spin_a = solution_.dimensionless_spin();
  const auto a_squared =
      std::inner_product(spin_a.begin(), spin_a.end(), spin_a.begin(), 0.);
  const auto& rho = get(cache->get_var(*this, internal_tags::rho<DataType>{}));
  const auto& r = get(cache->get_var(*this, internal_tags::r<DataType>{}));

  for (size_t i = 0; i < 3; ++i) {
    for (size_t m = 0; m < 3; ++m) {
      matrix_G1->get(i, m) = 1. / square(rho) / r;
    }
  }
  for (size_t i = 0; i < 3; ++i) {
    for (size_t m = 0; m < 3; ++m) {
      // Kronecker delta
      if (i == m) {
        matrix_G1->get(i, m) *=
            (a_squared - gsl::at(spin_a, i) * gsl::at(spin_a, m));
      } else {
        matrix_G1->get(i, m) *= -gsl::at(spin_a, i) * gsl::at(spin_a, m);
      }
    }
  }
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> a_dot_x,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::a_dot_x<DataType> /*meta*/) const {
  const auto& x_sph_minus_center = cache->get_var(
      *this, internal_tags::x_sph_minus_center<DataType, Frame>{});
  const auto spin_a = solution_.dimensionless_spin();

  get(*a_dot_x) = spin_a[0] * get<0>(x_sph_minus_center) +
                  spin_a[1] * get<1>(x_sph_minus_center) +
                  spin_a[2] * get<2>(x_sph_minus_center);
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> s_number,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::s_number<DataType> /*meta*/) const {
  const auto& r_squared =
      get(cache->get_var(*this, internal_tags::r_squared<DataType>{}));
  const auto& a_dot_x =
      get(cache->get_var(*this, internal_tags::a_dot_x<DataType>{}));

  get(*s_number) = r_squared + square(a_dot_x) / r_squared;
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> matrix_G2,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::matrix_G2<DataType, Frame> /*meta*/) const {
  const auto& matrix_Q =
      cache->get_var(*this, internal_tags::matrix_Q<DataType, Frame>{});
  const auto& rho = get(cache->get_var(*this, internal_tags::rho<DataType>{}));
  const auto& r = get(cache->get_var(*this, internal_tags::r<DataType>{}));
  const auto& s_number =
      get(cache->get_var(*this, internal_tags::s_number<DataType>{}));

  for (size_t n = 0; n < 3; ++n) {
    for (size_t j = 0; j < 3; ++j) {
      matrix_G2->get(n, j) = square(rho) / (s_number * r);
    }
  }
  for (size_t n = 0; n < 3; ++n) {
    for (size_t j = 0; j < 3; ++j) {
      matrix_G2->get(n, j) *= matrix_Q.get(n, j);
    }
  }
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3, Frame>*> G1_dot_x,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::G1_dot_x<DataType, Frame> /*meta*/) const {
  const auto& x_sph_minus_center = cache->get_var(
      *this, internal_tags::x_sph_minus_center<DataType, Frame>{});
  const auto& matrix_G1 =
      cache->get_var(*this, internal_tags::matrix_G1<DataType, Frame>{});

  for (size_t i = 0; i < 3; ++i) {
    G1_dot_x->get(i) = matrix_G1.get(i, 0) * x_sph_minus_center.get(0);
    for (size_t m = 1; m < 3; ++m) {
      G1_dot_x->get(i) += matrix_G1.get(i, m) * x_sph_minus_center.get(m);
    }
  }
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::i<DataType, 3, Frame>*> G2_dot_x,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::G2_dot_x<DataType, Frame> /*meta*/) const {
  const auto& x_sph_minus_center = cache->get_var(
      *this, internal_tags::x_sph_minus_center<DataType, Frame>{});
  const auto& matrix_G2 =
      cache->get_var(*this, internal_tags::matrix_G2<DataType, Frame>{});

  for (size_t j = 0; j < 3; ++j) {
    G2_dot_x->get(j) = matrix_G2.get(0, j) * x_sph_minus_center.get(0);
    for (size_t n = 1; n < 3; ++n) {
      G2_dot_x->get(j) += matrix_G2.get(n, j) * x_sph_minus_center.get(n);
    }
  }
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> inv_jacobian,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::inv_jacobian<DataType, Frame> /*meta*/) const {
  const auto& matrix_Q =
      cache->get_var(*this, internal_tags::matrix_Q<DataType, Frame>{});
  const auto& G1_dot_x =
      cache->get_var(*this, internal_tags::G1_dot_x<DataType, Frame>{});
  const auto& G2_dot_x =
      cache->get_var(*this, internal_tags::G2_dot_x<DataType, Frame>{});

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      inv_jacobian->get(j, i) =
          matrix_Q.get(i, j) + G1_dot_x.get(i) * G2_dot_x.get(j);
    }
  }
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> matrix_E1,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::matrix_E1<DataType, Frame> /*meta*/) const {
  const auto& rho = get(cache->get_var(*this, internal_tags::rho<DataType>{}));
  const auto& r_squared =
      get(cache->get_var(*this, internal_tags::r_squared<DataType>{}));
  const auto spin_a = solution_.dimensionless_spin();
  const auto a_squared =
      std::inner_product(spin_a.begin(), spin_a.end(), spin_a.begin(), 0.);

  for (size_t i = 0; i < 3; ++i) {
    for (size_t m = 0; m < 3; ++m) {
      matrix_E1->get(i, m) =
          -1. / square(rho) * (1. / r_squared + 2 / square(rho));
      // Kronecker delta
      if (i == m) {
        matrix_E1->get(i, m) *=
            (a_squared - gsl::at(spin_a, i) * gsl::at(spin_a, m));
      } else {
        matrix_E1->get(i, m) *= -gsl::at(spin_a, i) * gsl::at(spin_a, m);
      }
    }
  }
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> matrix_E2,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::matrix_E2<DataType, Frame> /*meta*/) const {
  const auto& rho = get(cache->get_var(*this, internal_tags::rho<DataType>{}));
  const auto spin_a = solution_.dimensionless_spin();
  const auto a_squared =
      std::inner_product(spin_a.begin(), spin_a.end(), spin_a.begin(), 0.);
  const auto& r = get(cache->get_var(*this, internal_tags::r<DataType>{}));
  const auto& a_dot_x =
      get(cache->get_var(*this, internal_tags::a_dot_x<DataType>{}));
  const auto& s_number =
      get(cache->get_var(*this, internal_tags::s_number<DataType>{}));
  const auto& matrix_G2 =
      cache->get_var(*this, internal_tags::matrix_G2<DataType, Frame>{});
  const auto& matrix_P =
      cache->get_var(*this, internal_tags::matrix_P<DataType, Frame>{});

  for (size_t n = 0; n < 3; ++n) {
    for (size_t j = 0; j < 3; ++j) {
      matrix_E2->get(n, j) = (1. / s_number) * matrix_P.get(n, j);
    }
  }
  for (size_t n = 0; n < 3; ++n) {
    for (size_t j = 0; j < 3; ++j) {
      matrix_E2->get(n, j) +=
          ((-a_squared / (square(rho) * r)) -
           (2. / s_number) * (r - (square(a_dot_x) / cube(r)))) *
          matrix_G2.get(n, j);
    }
  }
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::iJk<DataType, 3, Frame>*> deriv_inv_jacobian,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::deriv_inv_jacobian<DataType, Frame> /*meta*/) const {
  const auto spin_a = solution_.dimensionless_spin();
  const auto& matrix_D =
      cache->get_var(*this, internal_tags::matrix_D<DataType, Frame>{});
  const auto& matrix_G1 =
      cache->get_var(*this, internal_tags::matrix_G1<DataType, Frame>{});
  const auto& matrix_G2 =
      cache->get_var(*this, internal_tags::matrix_G2<DataType, Frame>{});
  const auto& matrix_E1 =
      cache->get_var(*this, internal_tags::matrix_E1<DataType, Frame>{});
  const auto& matrix_E2 =
      cache->get_var(*this, internal_tags::matrix_E2<DataType, Frame>{});
  const auto& x_sph_minus_center = cache->get_var(
      *this, internal_tags::x_sph_minus_center<DataType, Frame>{});
  const auto& a_dot_x =
      get(cache->get_var(*this, internal_tags::a_dot_x<DataType>{}));
  const auto& r = get(cache->get_var(*this, internal_tags::r<DataType>{}));
  const auto& G1_dot_x =
      cache->get_var(*this, internal_tags::G1_dot_x<DataType, Frame>{});
  const auto& G2_dot_x =
      cache->get_var(*this, internal_tags::G2_dot_x<DataType, Frame>{});
  const auto& s_number =
      get(cache->get_var(*this, internal_tags::s_number<DataType>{}));

  for (size_t k = 0; k < 3; ++k) {
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        deriv_inv_jacobian->get(k, j, i) =
            matrix_D.get(i, j) * x_sph_minus_center.get(k) +
            matrix_G1.get(i, k) * G2_dot_x.get(j) +
            matrix_G2.get(k, j) * G1_dot_x.get(i) -
            2. * a_dot_x * gsl::at(spin_a, k) / s_number / square(r) *
                G1_dot_x.get(i) * G2_dot_x.get(j);

        for (size_t m = 0; m < 3; ++m) {
          deriv_inv_jacobian->get(k, j, i) +=
              matrix_E1.get(i, m) * x_sph_minus_center.get(m) *
                  G2_dot_x.get(j) * x_sph_minus_center.get(k) / r +
              G1_dot_x.get(i) * x_sph_minus_center.get(k) *
                  x_sph_minus_center.get(m) * matrix_E2.get(m, j) / r;
        }
      }
    }
  }
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> H,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::H<DataType> /*meta*/) const {
  const auto& r = get(cache->get_var(*this, internal_tags::r<DataType>{}));
  const auto& a_dot_x =
      get(cache->get_var(*this, internal_tags::a_dot_x<DataType>{}));

  get(*H) = solution_.mass() * cube(r) / (pow(r, 4) + square(a_dot_x));
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3, Frame>*> x_kerr_schild,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::x_kerr_schild<DataType, Frame> /*meta*/) const {
  const auto spin_a = solution_.dimensionless_spin();
  const auto& x_sph_minus_center = cache->get_var(
      *this, internal_tags::x_sph_minus_center<DataType, Frame>{});
  const auto& a_dot_x =
      get(cache->get_var(*this, internal_tags::a_dot_x<DataType>{}));
  const auto& r = get(cache->get_var(*this, internal_tags::r<DataType>{}));
  const auto& rho = get(cache->get_var(*this, internal_tags::rho<DataType>{}));

  for (size_t i = 0; i < 3; ++i) {
    x_kerr_schild->get(i) = rho / r * x_sph_minus_center.get(i) -
                            gsl::at(spin_a, i) * a_dot_x / r / (rho + r);
  }
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3, Frame>*> a_cross_x,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::a_cross_x<DataType, Frame> /*meta*/) const {
  const auto spin_a = solution_.dimensionless_spin();
  const tnsr::I<DataType, 3, Frame>& x_kerr_schild =
      cache->get_var(*this, internal_tags::x_kerr_schild<DataType, Frame>{});

  auto spin_tensor = make_with_value<tnsr::i<DataType, 3, Frame>>(
      get_size(get_element(x_kerr_schild, 0)), 0.0);

  for (size_t s = 0; s < 3; ++s) {
    spin_tensor[s] = spin_a[s];
  }
  auto temp_cross_product = cross_product(spin_tensor, x_kerr_schild);
  for (size_t i = 0; i < 3; ++i) {
    a_cross_x->get(i) = temp_cross_product[i];
  }
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3, Frame>*> kerr_schild_l,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::kerr_schild_l<DataType, Frame> /*meta*/) const {
  const auto spin_a = solution_.dimensionless_spin();
  const auto& a_dot_x =
      get(cache->get_var(*this, internal_tags::a_dot_x<DataType>{}));
  const auto& r = get(cache->get_var(*this, internal_tags::r<DataType>{}));
  const auto& a_cross_x =
      cache->get_var(*this, internal_tags::a_cross_x<DataType, Frame>{});
  const auto& rho = get(cache->get_var(*this, internal_tags::rho<DataType>{}));
  const auto& x_kerr_schild =
      cache->get_var(*this, internal_tags::x_kerr_schild<DataType, Frame>{});

  const auto den = 1. / square(rho);
  const auto rboyer = r;

  for (int i = 0; i < 3; ++i) {
    kerr_schild_l->get(i) =
        den * (rboyer * x_kerr_schild.get(i) +
               a_dot_x * gsl::at(spin_a, i) / rboyer - a_cross_x.get(i));
  }
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::i<DataType, 4, Frame>*> sph_kerr_schild_l_lower,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::sph_kerr_schild_l_lower<DataType, Frame> /*meta*/) const {
  const auto& kerr_schild_l =
      cache->get_var(*this, internal_tags::kerr_schild_l<DataType, Frame>{});
  const auto& jacobian =
      cache->get_var(*this, internal_tags::jacobian<DataType, Frame>{});

  sph_kerr_schild_l_lower->get(0) = 1.;

  for (size_t j = 0; j < 3; ++j) {
    sph_kerr_schild_l_lower->get(j + 1) = 0.;

    for (size_t i = 0; i < 3; ++i) {
      sph_kerr_schild_l_lower->get(j + 1) +=
          jacobian.get(j, i) * kerr_schild_l.get(i);
    }
  }
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::I<DataType, 4, Frame>*> sph_kerr_schild_l_upper,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::sph_kerr_schild_l_upper<DataType, Frame> /*meta*/) const {
  const auto& kerr_schild_l =
      cache->get_var(*this, internal_tags::kerr_schild_l<DataType, Frame>{});
  const auto& inv_jacobian =
      cache->get_var(*this, internal_tags::inv_jacobian<DataType, Frame>{});

  sph_kerr_schild_l_upper->get(0) = -1.;  // this is l^t

  for (size_t j = 0; j < 3; ++j) {
    sph_kerr_schild_l_upper->get(j + 1) = 0.;

    for (size_t i = 0; i < 3; ++i) {
      sph_kerr_schild_l_upper->get(j + 1) +=
          inv_jacobian.get(i, j) * kerr_schild_l.get(i);
    }
  }
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::I<DataType, 4, Frame>*> deriv_H,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::deriv_H<DataType, Frame> /*meta*/) const {
  const auto& r = get(cache->get_var(*this, internal_tags::r<DataType>{}));
  const auto& a_dot_x =
      get(cache->get_var(*this, internal_tags::a_dot_x<DataType>{}));
  const auto& x_kerr_schild =
      cache->get_var(*this, internal_tags::x_kerr_schild<DataType, Frame>{});
  const auto& H = cache->get_var(*this, internal_tags::H<DataType>{});
  const auto& jacobian =
      cache->get_var(*this, internal_tags::jacobian<DataType, Frame>{});
  const auto spin_a = solution_.dimensionless_spin();

  deriv_H->get(0) = 0.;

  const auto rboyer = r;
  auto mass_datavector = make_with_value<Scalar<DataType>>(
      get_size(get_element(x_kerr_schild, 0)), solution_.mass());

  const auto drden =
      get(H) / get(mass_datavector);  // H has M as a factor, but dr does not.

  auto dr = make_with_value<tnsr::i<DataType, 3, Frame>>(
      get_size(get_element(x_kerr_schild, 0)), 0.0);
  for (size_t i = 0; i < 3; ++i) {
    dr[i] = drden * (x_kerr_schild.get(i) +
                     a_dot_x * gsl::at(spin_a, i) / square(rboyer));
  }

  const auto Hden = 1. / (pow(rboyer, 4) + square(a_dot_x));
  const auto fac = 3. / rboyer - 4. * cube(rboyer) * Hden;
  for (size_t i = 0; i < 3; ++i) {
    deriv_H->get(i + 1) =
        get(H) *
        (fac * dr[i] -
         2. * Hden * a_dot_x * gsl::at(spin_a, i));  // deriv_H in original KS
  }

  const auto deriv_H_x = deriv_H->get(1);
  const auto deriv_H_y = deriv_H->get(2);
  const auto deriv_H_z = deriv_H->get(3);

  for (size_t j = 0; j < 3; ++j) {
    deriv_H->get(j + 1) = jacobian.get(j, 0) * deriv_H_x +
                          jacobian.get(j, 1) * deriv_H_y +
                          jacobian.get(j, 2) * deriv_H_z;
  }  // deriv_H in Spherical KS
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> lapse_squared,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::lapse_squared<DataType> /*meta*/) const {
  const auto& H = get(cache->get_var(*this, internal_tags::H<DataType>{}));
  const auto& sph_kerr_schild_l_upper = cache->get_var(
      *this, internal_tags::sph_kerr_schild_l_upper<DataType, Frame>{});
  get(*lapse_squared) =
      1.0 / (1.0 + 2.0 * square(sph_kerr_schild_l_upper.get(0)) * H);
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> lapse,
    const gsl::not_null<CachedBuffer*> cache,
    gr::Tags::Lapse<DataType> /*meta*/) const {
  const auto& lapse_squared =
      get(cache->get_var(*this, internal_tags::lapse_squared<DataType>{}));
  get(*lapse) = sqrt(lapse_squared);
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::ij<DataType, 4, Frame>*> ks_deriv_l,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::ks_deriv_l<DataType, Frame> /*meta*/) const {
  const auto& x_kerr_schild =
      cache->get_var(*this, internal_tags::x_kerr_schild<DataType, Frame>{});
  const auto& r = get(cache->get_var(*this, internal_tags::r<DataType>{}));
  const auto& a_dot_x =
      get(cache->get_var(*this, internal_tags::a_dot_x<DataType>{}));
  const auto& rho = get(cache->get_var(*this, internal_tags::rho<DataType>{}));
  const auto spin_a = solution_.dimensionless_spin();
  const auto& kerr_schild_l =
      cache->get_var(*this, internal_tags::kerr_schild_l<DataType, Frame>{});
  const auto& H = cache->get_var(*this, internal_tags::H<DataType>{});

  for (size_t i = 0; i < 4; ++i) {
    ks_deriv_l->get(i, 0) = 0.;
    ks_deriv_l->get(0, i) = 0.;
  }

  const auto den = 1. / square(rho);
  const auto rboyer = r;
  auto mass_datavector = make_with_value<Scalar<DataType>>(
      get_size(get_element(x_kerr_schild, 0)), solution_.mass());

  const auto drden = get(H) / get(mass_datavector);
  auto dr = make_with_value<tnsr::i<DataType, 3, Frame>>(
      get_size(get_element(x_kerr_schild, 0)), 0.0);
  for (size_t i = 0; i < 3; ++i) {
    dr[i] = drden * (x_kerr_schild.get(i) +
                     a_dot_x * gsl::at(spin_a, i) / square(rboyer));
  }
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      ks_deriv_l->get(j + 1, i + 1) =
          den * ((x_kerr_schild.get(i) - 2. * rboyer * kerr_schild_l.get(i) -
                  a_dot_x * gsl::at(spin_a, i) / square(rboyer)) *
                     dr[j] +
                 gsl::at(spin_a, i) * gsl::at(spin_a, j) / rboyer);
      if (i == j) {
        ks_deriv_l->get(j + 1, i + 1) += den * rboyer;
      } else {  //  add den*epsilon^ijk a_k
        size_t k = (j + 1) % 3;
        if (k == i) {  // j+1 = i (cyclic), so choose minus sign
          ++k;
          k %= 3;  // and set k to be neither i nor j
          ks_deriv_l->get(j + 1, i + 1) -= den * gsl::at(spin_a, k);
        } else {  // i+1 = j (cyclic), so choose plus sign
          ks_deriv_l->get(j + 1, i + 1) += den * gsl::at(spin_a, k);
        }
      }
    }
  }
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::ij<DataType, 4, Frame>*> sks_deriv_l,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::sks_deriv_l<DataType, Frame> /*meta*/) const {
  const auto& kerr_schild_l =
      cache->get_var(*this, internal_tags::kerr_schild_l<DataType, Frame>{});
  const auto& jacobian =
      cache->get_var(*this, internal_tags::jacobian<DataType, Frame>{});
  const auto& deriv_jacobian =
      cache->get_var(*this, internal_tags::deriv_jacobian<DataType, Frame>{});
  const auto& ks_deriv_l =
      cache->get_var(*this, internal_tags::ks_deriv_l<DataType, Frame>{});

  for (size_t i = 0; i < 4; ++i) {
    sks_deriv_l->get(i, 0) = 0.;
    sks_deriv_l->get(0, i) = 0.;
  }

  for (size_t j = 0; j < 3; ++j) {
    for (size_t i = 0; i < 3; ++i) {
      sks_deriv_l->get(j + 1, i + 1) = 0.;
      for (size_t k = 0; k < 3; ++k) {
        for (size_t m = 0; m < 3; ++m) {
          sks_deriv_l->get(j + 1, i + 1) += jacobian.get(i, k) *
                                            jacobian.get(j, m) *
                                            ks_deriv_l.get(m + 1, k + 1);
        }
        sks_deriv_l->get(j + 1, i + 1) +=
            kerr_schild_l.get(k) * deriv_jacobian.get(j, i, k);
      }
    }
  }
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> deriv_lapse_multiplier,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::deriv_lapse_multiplier<DataType> /*meta*/) const {
  const auto& lapse = get(cache->get_var(*this, gr::Tags::Lapse<DataType>{}));
  const auto& lapse_squared =
      get(cache->get_var(*this, internal_tags::lapse_squared<DataType>{}));
  get(*deriv_lapse_multiplier) =
      -square(null_vector_0_) * lapse * lapse_squared;
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> shift_multiplier,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::shift_multiplier<DataType> /*meta*/) const {
  const auto& H = get(cache->get_var(*this, internal_tags::H<DataType>{}));
  const auto& lapse_squared =
      get(cache->get_var(*this, internal_tags::lapse_squared<DataType>{}));

  get(*shift_multiplier) = -2.0 * null_vector_0_ * H * lapse_squared;
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3, Frame>*> shift,
    const gsl::not_null<CachedBuffer*> cache,
    gr::Tags::Shift<3, Frame, DataType> /*meta*/) const {
  const auto& sph_kerr_schild_l_upper = cache->get_var(
      *this, internal_tags::sph_kerr_schild_l_upper<DataType, Frame>{});
  const auto& shift_multiplier =
      get(cache->get_var(*this, internal_tags::shift_multiplier<DataType>{}));

  for (size_t i = 0; i < 3; ++i) {
    shift->get(i) = shift_multiplier * sph_kerr_schild_l_upper.get(i + 1);
  }
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::iJ<DataType, 3, Frame>*> deriv_shift,
    const gsl::not_null<CachedBuffer*> cache,
    DerivShift<DataType, Frame> /*meta*/) const {
  const auto& H = get(cache->get_var(*this, internal_tags::H<DataType>{}));
  const auto& sph_kerr_schild_l_upper = cache->get_var(
      *this, internal_tags::sph_kerr_schild_l_upper<DataType, Frame>{});
  const auto& sph_kerr_schild_l_lower = cache->get_var(
      *this, internal_tags::sph_kerr_schild_l_lower<DataType, Frame>{});
  const auto& lapse_squared =
      get(cache->get_var(*this, internal_tags::lapse_squared<DataType>{}));
  const auto& deriv_H =
      cache->get_var(*this, internal_tags::deriv_H<DataType, Frame>{});
  const auto& sks_deriv_l =
      cache->get_var(*this, internal_tags::sks_deriv_l<DataType, Frame>{});
  const auto& inv_jacobian =
      cache->get_var(*this, internal_tags::inv_jacobian<DataType, Frame>{});
  const auto& deriv_inv_jacobian = cache->get_var(
      *this, internal_tags::deriv_inv_jacobian<DataType, Frame>{});

  for (int i = 0; i < 3; ++i) {
    for (int k = 0; k < 3; ++k) {
      deriv_shift->get(k, i) =
          4.0 * H * sph_kerr_schild_l_upper.get(0) *
              sph_kerr_schild_l_upper.get(i + 1) * square(lapse_squared) *
              (square(sph_kerr_schild_l_upper.get(0)) * deriv_H.get(k + 1) +
               2.0 * H * sph_kerr_schild_l_upper.get(0) *
                   sks_deriv_l.get(k + 1, 0)) -
          2.0 * lapse_squared *
              (sph_kerr_schild_l_upper.get(0) *
                   sph_kerr_schild_l_upper.get(i + 1) * deriv_H.get(k + 1) +
               H * sph_kerr_schild_l_upper.get(i + 1) *
                   sks_deriv_l.get(k + 1, 0));

      for (int j = 0; j < 3; ++j) {
        for (int m = 0; m < 3; ++m) {
          deriv_shift->get(k, i) +=
              -2.0 * lapse_squared * H * sph_kerr_schild_l_upper.get(0) *
              (inv_jacobian.get(j, i) * inv_jacobian.get(j, m) *
                   sks_deriv_l.get(k + 1, m + 1) +
               inv_jacobian.get(j, i) * sph_kerr_schild_l_lower.get(m + 1) *
                   deriv_inv_jacobian.get(k, j, m) +
               inv_jacobian.get(j, m) * sph_kerr_schild_l_lower.get(m + 1) *
                   deriv_inv_jacobian.get(k, j, i));
        }
      }
    }
  }
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::ii<DataType, 3, Frame>*> spatial_metric,
    const gsl::not_null<CachedBuffer*> cache,
    gr::Tags::SpatialMetric<3, Frame, DataType> /*meta*/) const {
  const auto& H = get(cache->get_var(*this, internal_tags::H<DataType>{}));
  const auto& sph_kerr_schild_l_lower = cache->get_var(
      *this, internal_tags::sph_kerr_schild_l_lower<DataType, Frame>{});
  const auto& jacobian =
      cache->get_var(*this, internal_tags::jacobian<DataType, Frame>{});

  std::fill(spatial_metric->begin(), spatial_metric->end(), 0.);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {  // Symmetry
      spatial_metric->get(i, j) += 2.0 * H *
                                   sph_kerr_schild_l_lower.get(i + 1) *
                                   sph_kerr_schild_l_lower.get(j + 1);
    }
  }

  for (size_t k = 0; k < 3; ++k) {
    for (size_t m = k; m < 3; ++m) {
      for (size_t i = 0; i < 3; ++i) {
        spatial_metric->get(k, m) += jacobian.get(k, i) * jacobian.get(m, i);
      }
    }
  }
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::ijj<DataType, 3, Frame>*> deriv_spatial_metric,
    const gsl::not_null<CachedBuffer*> cache,
    DerivSpatialMetric<DataType, Frame> /*meta*/) const {
  const auto& sph_kerr_schild_l_lower = cache->get_var(
      *this, internal_tags::sph_kerr_schild_l_lower<DataType, Frame>{});
  const auto& deriv_H =
      cache->get_var(*this, internal_tags::deriv_H<DataType, Frame>{});
  const auto& H = get(cache->get_var(*this, internal_tags::H<DataType>{}));
  const auto& sks_deriv_l =
      cache->get_var(*this, internal_tags::sks_deriv_l<DataType, Frame>{});
  const auto& jacobian =
      cache->get_var(*this, internal_tags::jacobian<DataType, Frame>{});
  const auto& deriv_jacobian =
      cache->get_var(*this, internal_tags::deriv_jacobian<DataType, Frame>{});

  for (int k = 0; k < 3; ++k) {
    for (int i = 0; i < 3; ++i) {
      for (int j = i; j < 3; ++j) {  // Symmetry
        deriv_spatial_metric->get(k, i, j) =
            2.0 * sph_kerr_schild_l_lower.get(i + 1) *
                sph_kerr_schild_l_lower.get(j + 1) * deriv_H.get(k + 1) +
            2.0 * H *
                (sph_kerr_schild_l_lower.get(i + 1) *
                     sks_deriv_l.get(k + 1, j + 1) +
                 sph_kerr_schild_l_lower.get(j + 1) *
                     sks_deriv_l.get(k + 1, i + 1));
        for (int m = 0; m < 3; ++m) {
          deriv_spatial_metric->get(k, i, j) +=
              deriv_jacobian.get(k, i, m) * jacobian.get(j, m) +
              deriv_jacobian.get(k, j, m) * jacobian.get(i, m);
        }
      }
    }
  }
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::ii<DataType, 3, Frame>*> dt_spatial_metric,
    const gsl::not_null<CachedBuffer*> /*cache*/,
    ::Tags::dt<gr::Tags::SpatialMetric<3, Frame, DataType>> /*meta*/) const {
  std::fill(dt_spatial_metric->begin(), dt_spatial_metric->end(), 0.);
}

template <typename DataType, typename Frame>
tnsr::i<DataType, 3, Frame>
SphKerrSchild::IntermediateVars<DataType, Frame>::get_var(
    const IntermediateComputer<DataType, Frame>& computer,
    DerivLapse<DataType, Frame> /*meta*/) {
  tnsr::i<DataType, 3, Frame> result{};
  const auto& deriv_H =
      get_var(computer, internal_tags::deriv_H<DataType, Frame>{});
  const auto& deriv_lapse_multiplier =
      get(get_var(computer, internal_tags::deriv_lapse_multiplier<DataType>{}));

  for (size_t i = 0; i < 3; ++i) {
    result.get(i) = deriv_lapse_multiplier * deriv_H.get(i + 1);
  }
  return result;
}

template <typename DataType, typename Frame>
Scalar<DataType> SphKerrSchild::IntermediateVars<DataType, Frame>::get_var(
    const IntermediateComputer<DataType, Frame>& computer,
    ::Tags::dt<gr::Tags::Lapse<DataType>> /*meta*/) {
  const auto& H = get(get_var(computer, internal_tags::H<DataType>{}));
  return make_with_value<Scalar<DataType>>(H, 0.);
}

template <typename DataType, typename Frame>
tnsr::I<DataType, 3, Frame>
SphKerrSchild::IntermediateVars<DataType, Frame>::get_var(
    const IntermediateComputer<DataType, Frame>& computer,
    ::Tags::dt<gr::Tags::Shift<3, Frame, DataType>> /*meta*/) {
  const auto& H = get(get_var(computer, internal_tags::H<DataType>()));
  return make_with_value<tnsr::I<DataType, 3, Frame>>(H, 0.);
}

template <typename DataType, typename Frame>
Scalar<DataType> SphKerrSchild::IntermediateVars<DataType, Frame>::get_var(
    const IntermediateComputer<DataType, Frame>& computer,
    gr::Tags::SqrtDetSpatialMetric<DataType> /*meta*/) {
  const auto& jacobian =
      get_var(computer, internal_tags::jacobian<DataType, Frame>{});

  auto det_jacobian = determinant(jacobian);
  return Scalar<DataType>(get(det_jacobian) /
                          get(get_var(computer, gr::Tags::Lapse<DataType>{})));
}

template <typename DataType, typename Frame>
tnsr::i<DataType, 3, Frame>
SphKerrSchild::IntermediateVars<DataType, Frame>::get_var(
    const IntermediateComputer<DataType, Frame>& computer,
    gr::Tags::DerivDetSpatialMetric<3, Frame, DataType> /*meta*/) {
  const auto& deriv_H =
      get_var(computer, internal_tags::deriv_H<DataType, Frame>{});

  auto result =
      make_with_value<tnsr::i<DataType, 3, Frame>>(get<0>(deriv_H), 0.);
  for (size_t i = 0; i < 3; ++i) {
    result.get(i) = 2.0 * square(null_vector_0_) * deriv_H.get(i + 1);
  }

  return result;
}

template <typename DataType, typename Frame>
tnsr::II<DataType, 3, Frame>
SphKerrSchild::IntermediateVars<DataType, Frame>::get_var(
    const IntermediateComputer<DataType, Frame>& computer,
    gr::Tags::InverseSpatialMetric<3, Frame, DataType> /*meta*/) {
  const auto& H = get(get_var(computer, internal_tags::H<DataType>{}));
  const auto& lapse_squared =
      get(get_var(computer, internal_tags::lapse_squared<DataType>{}));
  const auto& sph_kerr_schild_l_upper = get_var(
      computer, internal_tags::sph_kerr_schild_l_upper<DataType, Frame>{});
  const auto& inv_jacobian =
      get_var(computer, internal_tags::inv_jacobian<DataType, Frame>{});

  auto result = make_with_value<tnsr::II<DataType, 3, Frame>>(H, 0.);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {  // Symmetry
      result.get(i, j) -= 2.0 * H * lapse_squared *
                          sph_kerr_schild_l_upper.get(i + 1) *
                          sph_kerr_schild_l_upper.get(j + 1);
    }
  }

  for (size_t k = 0; k < 3; ++k) {
    for (size_t m = k; m < 3; ++m) {
      for (size_t i = 0; i < 3; ++i) {
        result.get(k, m) += inv_jacobian.get(i, k) * inv_jacobian.get(i, m);
      }
    }
  }
  return result;
}

template <typename DataType, typename Frame>
tnsr::ii<DataType, 3, Frame>
SphKerrSchild::IntermediateVars<DataType, Frame>::get_var(
    const IntermediateComputer<DataType, Frame>& computer,
    gr::Tags::ExtrinsicCurvature<3, Frame, DataType> /*meta*/) {
  return gr::extrinsic_curvature(
      get_var(computer, gr::Tags::Lapse<DataType>{}),
      get_var(computer, gr::Tags::Shift<3, Frame, DataType>{}),
      get_var(computer, DerivShift<DataType, Frame>{}),
      get_var(computer, gr::Tags::SpatialMetric<3, Frame, DataType>{}),
      get_var(computer,
              ::Tags::dt<gr::Tags::SpatialMetric<3, Frame, DataType>>{}),
      get_var(computer, DerivSpatialMetric<DataType, Frame>{}));
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                \
  template class SphKerrSchild::IntermediateVars<DTYPE(data), FRAME(data)>; \
  template class SphKerrSchild::IntermediateComputer<DTYPE(data), FRAME(data)>;
GENERATE_INSTANTIATIONS(INSTANTIATE, (DataVector, double),
                        (::Frame::Inertial, ::Frame::Grid))
#undef INSTANTIATE
#undef DTYPE
#undef FRAME
}  // namespace gr::Solutions

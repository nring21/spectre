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
    r_squared->get() += square(x_sph_minus_center.get(i));

    std::cout << "this is r^2:" << r_squared << "\n";
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

// template <typename DataType, typename Frame>
// void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
//     const gsl::not_null<Scalar<DataType>*> a_dot_x,
//     const gsl::not_null<CachedBuffer*> cache,
//     internal_tags::a_dot_x<DataType> /*meta*/) const noexcept {
//   const auto& x_sph_minus_center =
//       cache->get_var(internal_tags::x_sph_minus_center<DataType, Frame>{});

//   const auto spin_a = solution_.dimensionless_spin() * solution_.mass();
//   get(*a_dot_x) = spin_a[0] * get<0>(x_sph_minus_center) +
//                   spin_a[1] * get<1>(x_sph_minus_center) +
//                   spin_a[2] * get<2>(x_sph_minus_center);
// }

// template <typename DataType, typename Frame>
// void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
//     const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> matrix_F,
//     const gsl::not_null<CachedBuffer*> cache,
//     internal_tags::matrix_F<DataType, Frame> /*meta*/) const noexcept {
//   const auto spin_a = solution_.dimensionless_spin() * solution_.mass();
//   const auto a_squared =
//       std::inner_product(spin_a.begin(), spin_a.end(), spin_a.begin(), 0.);
//   const auto& rho = get(cache->get_var(internal_tags::rho<DataType>{}));
//   const auto& r = get(cache->get_var(internal_tags::r<DataType>{}));

//   for (int i = 0; i < 3; ++i) {
//     for (int j = 0; j < 3; ++j) {
//       matrix_F->get(i, j) = -1. / rho / cube(r);
//     }
//   }
//   // F matrix
//   for (size_t i = 0; i < 3; ++i) {
//     for (size_t j = 0; j < 3; ++j) {
//       if (i == j) {
//         matrix_F->get(i, j) *= (a_squared - spin_a[i] * spin_a[j]);
//       } else {
//         matrix_F->get(i, j) *= -spin_a[i] * spin_a[j];
//       }
//     }
//   }
// }

// // TEST
// template <typename DataType, typename Frame>
// void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
//     const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> matrix_P,
//     const gsl::not_null<CachedBuffer*> cache,
//     internal_tags::matrix_P<DataType, Frame> /*meta*/) const noexcept {
//   const auto spin_a = solution_.dimensionless_spin() * solution_.mass();
//   const auto& rho = get(cache->get_var(internal_tags::rho<DataType>{}));
//   const auto& r = get(cache->get_var(internal_tags::r<DataType>{}));

//   for (size_t i = 0; i < 3; ++i) {
//     for (size_t j = 0; j < 3; ++j) {
//       matrix_P->get(i, j) = -1. / (rho + r) / r;
//     }
//   }
//   // P matrix
//   for (size_t i = 0; i < 3; ++i) {
//     for (size_t j = 0; j < 3; ++j) {
//       if (i == j) {
//         matrix_P->get(i, j) *= spin_a[i] * spin_a[j];
//         matrix_P->get(i, j) += rho / r;
//       } else {
//         matrix_P->get(i, j) *= spin_a[i] * spin_a[j];
//       }
//     }
//   }
// }

// // TEST
// template <typename DataType, typename Frame>
// void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
//     const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> jacobian,
//     const gsl::not_null<CachedBuffer*> cache,
//     internal_tags::jacobian<DataType, Frame> /*meta*/) const noexcept {
//   const auto& x_sph_minus_center =
//       cache->get_var(internal_tags::x_sph_minus_center<DataType, Frame>{});
//   const auto& matrix_F =
//       cache->get_var(internal_tags::matrix_F<DataType, Frame>{});
//   const auto& matrix_P =
//       cache->get_var(internal_tags::matrix_P<DataType, Frame>{});

//   // Jacobian
//   for (size_t i = 0; i < 3; ++i) {
//     for (size_t j = 0; j < 3; ++j) {
//       jacobian->get(i, j) = get<i,j>(matrix_P);
//       for (size_t k = 0; k < 3; ++k) {
//         jacobian->get(i, j) += get<i,k>(matrix_F) *
//         get<k>(x_sph_minus_center) *
//                                get<j>(x_sph_minus_center);
//       }
//     }
//   }
// }

// // TEST
// template <typename DataType, typename Frame>
// void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
//     const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> matrix_D,
//     const gsl::not_null<CachedBuffer*> cache,
//     internal_tags::matrix_D<DataType, Frame> /*meta*/) const noexcept {
//   const auto spin_a = solution_.dimensionless_spin() * solution_.mass();
//   const auto a_squared =
//       std::inner_product(spin_a.begin(), spin_a.end(), spin_a.begin(), 0.);
//   const auto& rho = get(cache->get_var(internal_tags::rho<DataType>{}));
//   const auto& r = get(cache->get_var(internal_tags::r<DataType>{}));

//   for (size_t i = 0; i < 3; ++i) {
//     for (size_t j = 0; j < 3; ++j) {
//       matrix_D->get(i, j) = 1. / cube(rho) / r;
//     }
//   }
//   // D matrix
//   for (size_t i = 0; i < 3; ++i) {
//     for (size_t j = i; j < 3; ++j) {
//       if (i == j) {
//         matrix_D->get(i, j) *= (a_squared - spin_a[i] * spin_a[j]);
//       } else {
//         matrix_D->get(i, j) *= -spin_a[i] * spin_a[j];
//       }
//     }
//   }
// }

// // TEST
// template <typename DataType, typename Frame>
// void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
//     const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> matrix_C,
//     const gsl::not_null<CachedBuffer*> cache,
//     internal_tags::matrix_C<DataType, Frame> /*meta*/) const noexcept {
//   const auto& matrix_F =
//       cache->get_var(internal_tags::matrix_F<DataType, Frame>{});
//   const auto& matrix_D =
//       cache->get_var(internal_tags::matrix_D<DataType, Frame>{});

//   // C matrix
//   for (size_t i = 0; i < 3; ++i) {
//     for (size_t j = 0; j < 3; ++j) {
//       matrix_C->get(i, j) = matrix_D.get(i, j) - 3. * matrix_F.get(i, j);
//     }
//   }
// }

// // TEST
// template <typename DataType, typename Frame>
// void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
//     const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> deriv_jacobian,
//     const gsl::not_null<CachedBuffer*> cache,
//     internal_tags::deriv_jacobian<DataType, Frame> /*meta*/) const noexcept {
//   const auto& matrix_C =
//       cache->get_var(internal_tags::matrix_C<DataType, Frame>{});
//   const auto& matrix_F =
//       cache->get_var(internal_tags::matrix_F<DataType, Frame>{});
//   const auto& x_sph_minus_center =
//       cache->get_var(internal_tags::x_sph_minus_center<DataType, Frame>{});
//   const auto& r_squared =
//       get(cache->get_var(internal_tags::r_squared<DataType>{}));

//   // deriv_Jacobian
//   for (size_t k = 0; k < 3; ++k) {
//     for (size_t i = 0; i < 3; ++i) {
//       for (size_t j = 0; j < 3; ++j) {
//         deriv_jacobian->get(i, j) =
//             matrix_F.get(i, j) * x_sph_minus_center.get(k) +
//             matrix_F.get(i, k) * x_sph_minus_center.get(j);
//         for (size_t m = 0; m < 3; ++m) {
//           if (j == k) {  // j==k acts as a Kronecker delta
//             deriv_jacobian->get(i, j) +=
//                 matrix_F.get(i, m) * x_sph_minus_center.get(m);
//           }
//           deriv_jacobian->get(i, j) +=
//               matrix_C.get(i, m) * x_sph_minus_center.get(k) *
//               x_sph_minus_center.get(m) * x_sph_minus_center.get(j) /
//               r_squared;
//         }
//       }
//     }
//   }
// }

// // TEST
// template <typename DataType, typename Frame>
// void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
//     const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> matrix_Q,
//     const gsl::not_null<CachedBuffer*> cache,
//     internal_tags::matrix_Q<DataType, Frame> /*meta*/) const noexcept {
//   const auto spin_a = solution_.dimensionless_spin() * solution_.mass();
//   const auto& rho = get(cache->get_var(internal_tags::rho<DataType>{}));
//   const auto& r = get(cache->get_var(internal_tags::r<DataType>{}));

//   // Q matrix
//   for (size_t i = 0; i < 3; ++i) {
//     for (size_t j = 0; j < 3; ++j) {
//       matrix_Q->get(i, j) = 1. / (rho + r) / rho;
//       if (i == j) {
//         matrix_Q->get(i, j) *= spin_a[i] * spin_a[j];
//         matrix_Q->get(i, j) += r / rho;
//       } else {
//         matrix_Q->get(i, j) *= spin_a[i] * spin_a[j];
//       }
//     }
//   }
// }

// // TEST
// template <typename DataType, typename Frame>
// void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
//     const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> matrix_G1,
//     const gsl::not_null<CachedBuffer*> cache,
//     internal_tags::matrix_G1<DataType, Frame> /*meta*/) const noexcept {
//   const auto spin_a = solution_.dimensionless_spin() * solution_.mass();
//   const auto a_squared =
//       std::inner_product(spin_a.begin(), spin_a.end(), spin_a.begin(), 0.);
//   const auto& rho = get(cache->get_var(internal_tags::rho<DataType>{}));
//   const auto& r = get(cache->get_var(internal_tags::r<DataType>{}));

//   for (size_t i = 0; i < 3; ++i) {
//     for (size_t j = i; j < 3; ++j) {
//       matrix_G1->get(i, j) = 1. / square(rho) / r;
//       if (i == j) {
//         matrix_G1->get(i, j) *= (a_squared - spin_a[i] * spin_a[j]);
//       } else {
//         matrix_G1->get(i, j) *= -spin_a[i] * spin_a[j];
//       }
//     }
//   }
// }

// // ADD TO HPP
// template <typename DataType, typename Frame>
// void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
//     const gsl::not_null<Scalar<DataType>*> s_number,
//     const gsl::not_null<CachedBuffer*> cache,
//     internal_tags::s_number<DataType> /*meta*/) const noexcept {
//   const auto& r_squared =
//       get(cache->get_var(internal_tags::r_squared<DataType>{}));
//   const auto& a_dot_x =
//   get(cache->get_var(internal_tags::a_dot_x<DataType>{}));

//   get(*s_number) = r_squared + square(a_dot_x) / r_squared;
// }

// // TEST
// template <typename DataType, typename Frame>
// void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
//     const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> matrix_G2,
//     const gsl::not_null<CachedBuffer*> cache,
//     internal_tags::matrix_G2<DataType, Frame> /*meta*/) const noexcept {
//   const auto& matrix_Q =
//       cache->get_var(internal_tags::matrix_Q<DataType, Frame>{});
//   const auto& rho = get(cache->get_var(internal_tags::rho<DataType>{}));
//   const auto& r = get(cache->get_var(internal_tags::r<DataType>{}));
//   const auto& s_number =
//       get(cache->get_var(internal_tags::s_number<DataType>{}));

//   for (size_t i = 0; i < 3; ++i) {
//     for (size_t j = i; j < 3; ++j) {
//       matrix_G2->get(i, j) = (square(rho) / r) * matrix_Q.get(i, j) /
//       s_number;
//     }
//   }
// }

// template <typename DataType, typename Frame>
// void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
//     const gsl::not_null<Scalar<DataType>*> G1_dot_x,
//     const gsl::not_null<CachedBuffer*> cache,
//     internal_tags::G1_dot_x<DataType> /*meta*/) const noexcept {
//   const auto& x_sph_minus_center =
//       cache->get_var(internal_tags::x_sph_minus_center<DataType, Frame>{});

//   get(*s_number) = r_squared + square(a_dot_x) / r_squared;
// }

// // TEST
// template <typename DataType, typename Frame>
// void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
//     const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> inv_jacobian,
//     const gsl::not_null<CachedBuffer*> cache,
//     internal_tags::inv_jacobian<DataType, Frame> /*meta*/) const noexcept {
//   const auto& matrix_Q =
//       cache->get_var(internal_tags::matrix_Q<DataType, Frame>{});
//   const auto& matrix_G1 =
//       cache->get_var(internal_tags::matrix_G1<DataType, Frame>{});
//   const auto& matrix_G2 =
//       cache->get_var(internal_tags::matrix_G2<DataType, Frame>{});
//   const auto& x_sph_minus_center =
//       cache->get_var(internal_tags::x_sph_minus_center<DataType, Frame>{});

//   tnsr::I<DataVector, 3, Frame> G1_dot_x{3, 0.};
//   tnsr::I<DataVector, 3, Frame> G2_dot_x{3, 0.};
//   tnsr::Ij<DataVector, 3, Frame> matrix_G{9, 0.};

//   // G1dotx^i: G1^i_m xhat^m
//   // G2dotx_j: G2^n_j xhat_n
//   for (size_t i = 0; i < 3; ++i) {
//     for (size_t m = 0; m < 3; ++m) {
//       G1_dot_x[i] += matrix_G1.get(i, m) * x_sph_minus_center.get(m);
//       G2_dot_x[i] += matrix_G2.get(m, i) * x_sph_minus_center.get(m);
//     }
//   }
//   for (size_t i = 0; i < 3; ++i) {
//     for (size_t j = 0; j < 3; ++j) {
//       matrix_G[i][j] = 1.0;
//     }
//   }

//   // InvJacobian
//   for (size_t i = 0; i < 3; ++i) {
//     for (size_t j = 0; j < 3; ++j) {
//       inv_jacobian->get(i, j) = matrix_Q.get(i, j) + matrix_G[i][j];
//     }
//   }
// }

// missing matrix_E1, matrix_E2, deriv_inv_jacobian ???

// // TEST
// template <typename DataType, typename Frame>
// void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
//     const gsl::not_null<tnsr::I<DataType, 3, Frame>*> x_kerr_schild,
//     const gsl::not_null<CachedBuffer*> cache,
//     internal_tags::x_kerr_schild<DataType, Frame> /*meta*/) const noexcept {
//   const auto spin_a = solution_.dimensionless_spin() * solution_.mass();
//   const auto& x_sph_minus_center =
//       cache->get_var(internal_tags::x_sph_minus_center<DataType, Frame>{});
//   const auto& a_dot_x =
//   get(cache->get_var(internal_tags::a_dot_x<DataType>{})); const auto& r =
//   get(cache->get_var(internal_tags::r<DataType>{})); const auto& rho =
//   get(cache->get_var(internal_tags::rho<DataType>{}));

//   for (size_t i = 0; i < 3; ++i) {
//     x_kerr_schild->get(i) = rho / r * x_sph_minus_center.get(i) -
//                             spin_a[i] * a_dot_x / r / (rho + r);
//   }
// }

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data) \
  template class SphKerrSchild::IntermediateComputer<DTYPE(data), FRAME(data)>;
GENERATE_INSTANTIATIONS(INSTANTIATE, (DataVector, double),
                        (::Frame::Inertial, ::Frame::Grid))
#undef INSTANTIATE
#undef DTYPE
#undef FRAME
}  // namespace gr::Solutions

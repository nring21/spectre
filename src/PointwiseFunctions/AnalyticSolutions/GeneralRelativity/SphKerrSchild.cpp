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

}  // namespace gr::Solutions
// Distributed under the MIT License.
// See LICENSE.txt for details

#include "NumericalAlgorithms/Spectral/SwshCoefficients.hpp"

#include <array>
#include <ostream>
#include <sharp_cxx.h>
#include <utility>

#include "ErrorHandling/Error.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace Spectral {
namespace Swsh {

CoefficientsMetadata::CoefficientsMetadata(const size_t l_max) noexcept
    : l_max_(l_max) {
  sharp_alm_info* alm_to_initialize;
  sharp_make_triangular_alm_info(l_max, l_max, 1, &alm_to_initialize);
  alm_info_.reset(alm_to_initialize);
}

namespace {
template <size_t I>
const CoefficientsMetadata& coefficients_cache_impl() noexcept {
  static const CoefficientsMetadata precomputed_coefficients{I};
  return precomputed_coefficients;
}

template <size_t... Is>
SPECTRE_ALWAYS_INLINE const CoefficientsMetadata&
precomputed_static_coefficients_impl(
    const size_t index, std::index_sequence<Is...> /*meta*/) noexcept {
  if (UNLIKELY(index > collocation_maximum_l_max)) {
    ERROR("The provided l_max "
          << index
          << " is not below the maximum l_max to cache, which is currently "
          << detail::coefficients_maximum_l_max
          << ". Either "
             "construct the CoefficientsMetadata manually, or consider (with "
             "caution) "
             "increasing `coefficients_maximum_l_max`.");
  }

  static const std::array<const CoefficientsMetadata& (*)(), sizeof...(Is)>
      cache{{&coefficients_cache_impl<Is>...}};
  return gsl::at(cache, index)();
}

}  // namespace

const CoefficientsMetadata& cached_coefficients_metadata(
    const size_t l_max) noexcept {
  return precomputed_static_coefficients_impl(
      l_max, std::make_index_sequence<detail::coefficients_maximum_l_max>{});
}
}  // namespace Swsh
}  // namespace Spectral

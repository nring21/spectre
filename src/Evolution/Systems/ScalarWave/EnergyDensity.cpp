// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarWave/EnergyDensity.hpp"

#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace ScalarWave {
template <size_t SpatialDim>
void energy_density(
    const gsl::not_null<Scalar<DataVector>*> result,
    const Scalar<DataVector>& pi,
    const tnsr::i<DataVector, SpatialDim, Frame::Inertial>& phi) noexcept {
  get(*result) +=
      0.5 * (get(pi) * get(pi) + get(magnitude(phi)) * get(magnitude(phi)));
}
}  // namespace ScalarWave

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                           \
  template void ScalarWave::energy_density(            \
      const gsl::not_null<Scalar<DataVector>*> result, \
      const Scalar<DataVector>& pi,                    \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>& phi) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM

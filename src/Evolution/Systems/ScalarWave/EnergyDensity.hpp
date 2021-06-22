// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"

namespace ScalarWave {

template <size_t SpatialDim>
void energy_density(
    const gsl::not_null<Scalar<DataVector>*> result,
    const Scalar<DataVector>& pi,
    const tnsr::i<DataVector, SpatialDim, Frame::Inertial>& phi) noexcept;

}  // namespace ScalarWave

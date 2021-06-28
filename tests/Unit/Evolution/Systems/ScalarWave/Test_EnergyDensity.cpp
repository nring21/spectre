// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <random>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/ScalarWave/EnergyDensity.hpp"
#include "Evolution/Systems/ScalarWave/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <size_t SpatialDim>
void test_energy_density(const Scalar<DataVector>& used_for_size) {
  void (*f)(const gsl::not_null<Scalar<DataVector>*>, const Scalar<DataVector>&,
            const tnsr::i<DataVector, SpatialDim, Frame::Inertial>&) =
      &ScalarWave::energy_density<SpatialDim>;
  pypp::check_with_random_values<1>(f, "EnergyDensity", {"energy_density"},
                                    {{{-1., 1.}}}, used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ScalarWave.EnergyDensity",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "Evolution/Systems/ScalarWave/");

  const DataVector used_for_size{3., 4., 5.};
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist(-1., 1.);

  const auto sdv = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&generator), dist, used_for_size);

  test_energy_density<1>(sdv);
}

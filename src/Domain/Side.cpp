// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Side.hpp"
#include "ErrorHandling/Error.hpp"

#include <ostream>

std::ostream& operator<<(std::ostream& os, const Side& side) {
  switch (side) {
    case Side::Lower:
      os << "Lower";
      break;
    case Side::Upper:
      os << "Upper";
      break;
    default:
      ERROR(
          "A Side that was neither Upper nor Lower was passed to the stream "
          "operator.");
  }
  return os;
}

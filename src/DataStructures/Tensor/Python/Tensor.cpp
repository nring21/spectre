// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/Tensor/Python/Tensor.hpp"

#include <memory>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>
#include <string>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PythonBindings/BoundChecks.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/GetOutput.hpp"

namespace py = pybind11;

namespace py_bindings {

namespace {

template <typename DataType>
std::string dtype_to_name() {
  if constexpr (std::is_same_v<DataType, DataVector>) {
    return "DV";
  } else if constexpr (std::is_same_v<DataType, double>) {
    return "D";
  } else {
    return "";
  }
}

// Used to generate Python class names for Tensor types
enum class TensorKind { Scalar, Tnsr, Jacobian };

template <typename TensorType, TensorKind Kind>
std::string class_name(const std::string& name) {
  if constexpr (Kind == TensorKind::Scalar) {
    return name + dtype_to_name<typename TensorType::type>();
  } else if constexpr (Kind == TensorKind::Tnsr) {
    return "Tensor" + name + dtype_to_name<typename TensorType::type>() +
           std::to_string(TensorType::index_dim(0)) +
           get_output(get<0>(TensorType::index_frames()));
  } else if constexpr (Kind == TensorKind::Jacobian) {
    // Jacobians have different frames, so handle them separately to other
    // Tensor instantiations where all indices are in the same frame
    return name + dtype_to_name<typename TensorType::type>() +
           std::to_string(TensorType::index_dim(0)) +
           get_output(get<1>(TensorType::index_frames())) + "To" +
           get_output(get<0>(TensorType::index_frames()));
  }
}

template <typename TensorType, TensorKind Kind>
void bind_tensor_impl(py::module& m, const std::string& name) {  // NOLINT
  auto tensor =
      py::class_<TensorType>(m, class_name<TensorType, Kind>(name).c_str(),
                             py::buffer_protocol())
          .def_property_readonly_static(
              "rank",
              [](const py::object& /*t*/) { return TensorType::rank(); })
          .def_property_readonly_static(
              "size",
              [](const py::object& /*t*/) { return TensorType::size(); })
          .def(py::init())
          .def("__str__", [](const TensorType& t) { return get_output(t); })
          .def(
              "__iter__",
              [](const TensorType& t) {
                return py::make_iterator(t.begin(), t.end());
              },
              // Keep object alive while iterator exists
              py::keep_alive<0, 1>())
          .def("__len__", [](const TensorType& t) { return t.size(); })
          .def("__getitem__",
               [](const TensorType& t, const size_t i) {
                 bounds_check(t, i);
                 return t[i];
               })
          .def("__setitem__",
               [](TensorType& t, const size_t i,
                  const typename TensorType::type& v) {
                 bounds_check(t, i);
                 t[i] = v;
               })
          .def(
              "multiplicity",
              [](const TensorType& t, const size_t storage_index) {
                return t.multiplicity(storage_index);
              },
              py::arg("storage_index"))
          .def(
              "component_suffix",
              [](const TensorType& t, const size_t storage_index) {
                return t.component_suffix(storage_index);
              },
              py::arg("storage_index"))
          // NOLINTNEXTLINE(misc-redundant-expression)
          .def(py::self == py::self)
          // NOLINTNEXTLINE(misc-redundant-expression)
          .def(py::self != py::self);

  if constexpr (std::is_same_v<typename TensorType::type, DataVector>) {
    tensor.def(py::init<size_t>(), py::arg("num_points"))
        .def(py::init<size_t, double>(), py::arg("num_points"), py::arg("fill"))
        // Support Python buffer protocol to cast to and from Numpy arrays
        .def(py::init([](const py::buffer& buffer, const bool copy) {
               py::buffer_info info = buffer.request();
               // Sanity-check the buffer
               if (info.format != py::format_descriptor<double>::format()) {
                 throw std::runtime_error(
                     "Incompatible format: expected a double array.");
               }
               if (info.ndim != 2) {
                 throw std::runtime_error(
                     "Tensor data is expected to be 2D with shape (size, "
                     "num_points).");
               }
               const auto size = static_cast<size_t>(info.shape[0]);
               if (size != TensorType::size()) {
                 throw std::runtime_error(
                     "This tensor type has " +
                     std::to_string(TensorType::size()) +
                     " independent components, but data has first dimension " +
                     std::to_string(size) + ".");
               }
               const auto num_points = static_cast<size_t>(info.shape[1]);
               auto data = static_cast<double*>(info.ptr);
               if (copy) {
                 TensorType result{num_points};
                 for (size_t i = 0; i < size; ++i) {
                   // NOLINTNEXTLINE
                   std::copy_n(data + i * num_points, num_points,
                               result[i].data());
                 }
                 return result;
               } else {
                 TensorType result{};
                 for (size_t i = 0; i < size; ++i) {
                   // NOLINTNEXTLINE
                   result[i].set_data_ref(data + i * num_points, num_points);
                 }
                 return result;
               }
             }),
             py::arg("buffer"), py::arg("copy") = true);
  } else if constexpr (std::is_same_v<typename TensorType::type, double>) {
    tensor.def(py::init<double>(), py::arg("fill"));
  }

  if constexpr (TensorType::rank() <= 1) {
    tensor.def(py::init<typename TensorType::storage_type>());
  }
}
}  // namespace

void bind_tensor(py::module& m) {
  bind_tensor_impl<Scalar<DataVector>, TensorKind::Scalar>(m, "Scalar");
  bind_tensor_impl<Scalar<double>, TensorKind::Scalar>(m, "Scalar");

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE_TNSR(_, data)                                  \
  bind_tensor_impl<tnsr::i<DTYPE(data), DIM(data), FRAME(data)>,   \
                   TensorKind::Tnsr>(m, "i");                      \
  bind_tensor_impl<tnsr::I<DTYPE(data), DIM(data), FRAME(data)>,   \
                   TensorKind::Tnsr>(m, "I");                      \
  bind_tensor_impl<tnsr::ij<DTYPE(data), DIM(data), FRAME(data)>,  \
                   TensorKind::Tnsr>(m, "ij");                     \
  bind_tensor_impl<tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>,  \
                   TensorKind::Tnsr>(m, "ii");                     \
  bind_tensor_impl<tnsr::II<DTYPE(data), DIM(data), FRAME(data)>,  \
                   TensorKind::Tnsr>(m, "II");                     \
  bind_tensor_impl<tnsr::ijj<DTYPE(data), DIM(data), FRAME(data)>, \
                   TensorKind::Tnsr>(m, "ijj");
#define INSTANTIATE_JAC(_, data)                                            \
  bind_tensor_impl<                                                         \
      Jacobian<DTYPE(data), DIM(data), Frame::ElementLogical, FRAME(data)>, \
      TensorKind::Jacobian>(m, "Jacobian");                                 \
  bind_tensor_impl<InverseJacobian<DTYPE(data), DIM(data),                  \
                                   Frame::ElementLogical, FRAME(data)>,     \
                   TensorKind::Jacobian>(m, "Jacobian");

  GENERATE_INSTANTIATIONS(INSTANTIATE_TNSR, (double, DataVector), (1, 2, 3),
                          (Frame::ElementLogical, Frame::Inertial))
  GENERATE_INSTANTIATIONS(INSTANTIATE_JAC, (double, DataVector), (1, 2, 3),
                          (Frame::Inertial))

#undef INSTANTIATE_TNSR
#undef INSTANTIATE_JAC
#undef DTYPE
#undef DIM
#undef FRAME
}

}  // namespace py_bindings

#include "pybind11/pybind11.h"

#include "xtensor/xmath.hpp"
#include "xtensor/xarray.hpp"

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"
#include "xtensor-python/pyvectorize.hpp"

#include <iostream>
#include <numeric>
#include <cmath>


#include <algorithm>
#include <cstdint>
#include <vector>
#include <iostream>


namespace py = pybind11;


template<class PTR_IN, class PTR_OUT>
void get_one_hot(
    const PTR_IN * labels,
    const uint64_t size,
    const uint64_t n_labels,
    PTR_OUT * one_hot
){
    std::fill(one_hot, one_hot+size*n_labels,0);
    for(uint64_t i=0; i<size; ++i)
    {   
        const auto l = labels[i];
        one_hot[n_labels*i + l] = 1;
    }
}



template<class T>
inline xt::pytensor<uint8_t,2 > py_one_hot(const xt::pytensor<T,1 > & labels, const uint64_t n_labels)
{
    auto one_hot = xt::pytensor<uint8_t,2 >::from_shape({ int64_t(labels.size()), int64_t(n_labels)});
    get_one_hot(labels.data(), labels.size(), n_labels, one_hot.data());
    return one_hot;
}


// Python Module and Docstrings

PYBIND11_MODULE(one_hot, m)
{
    xt::import_numpy();

    m.doc() = R"pbdoc(
        one hot encodings

        .. currentmodule:: one_hot

        .. autosummary::
           :toctree: _generate

           one_hot
    )pbdoc";

    m.def("one_hot", py_one_hot<uint8_t>, "one hot");
    m.def("one_hot", py_one_hot<int64_t>, "one hot");

}

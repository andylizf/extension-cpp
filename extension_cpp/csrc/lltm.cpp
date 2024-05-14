#include <torch/extension.h>

#include "cuda/lltm_cuda.h"

// Registers _C as an extension module.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("lltm_forward", &lltm_forward);
    m.def("lltm_backward", &lltm_backward);
}

// Defines the operators
TORCH_LIBRARY(TORCH_EXTENSION_NAME, m)
{
    m.def("lltm_forward", &lltm_forward);
    m.def("lltm_backward", &lltm_backward);
}

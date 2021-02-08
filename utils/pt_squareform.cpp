#include <torch/extension.h>
#include <iostream>

torch::Tensor square_to_vec(torch::Tensor M, torch::Tensor v, int n) {
  auto torch::Tensor *it;
  const double *cit;
  int i, j;
  auto s = torch::sigmoid(M);
  it = v;
  for (i = 0; i < n - 1; i++) {
    cit = M + (i * n) + i + 1;
    for (j = i + 1; j < n; j++, it++, cit++) {
      *it = *cit;
    }
  }

  return s;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("square_to_vec", &square_to_vec, "square to vec");
}
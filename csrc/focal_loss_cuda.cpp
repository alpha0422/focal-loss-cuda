#include <torch/torch.h>

#include <vector>

// CUDA forward declarations

at::Tensor focal_loss_forward_cuda(const at::Tensor &cls_output,
                                   const at::Tensor &cls_targets_at_level,
                                   const int64_t num_positives_sum,
                                   const int64_t num_classes, const float alpha,
                                   const float gamma,
                                   const float smoothing_factor);

at::Tensor focal_loss_backward_cuda(const at::Tensor &grad_output,
                                    const at::Tensor &cls_output,
                                    const at::Tensor &cls_targets_at_level,
                                    const int64_t num_positives_sum,
                                    const int64_t num_classes,
                                    const float alpha, const float gamma,
                                    const float smoothing_factor);

// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

at::Tensor focal_loss_forward(const at::Tensor &cls_output,
                              const at::Tensor &cls_targets_at_level,
                              const int64_t num_positives_sum,
                              const int64_t num_classes, const float alpha,
                              const float gamma, const float smoothing_factor) {
  CHECK_INPUT(cls_output);
  CHECK_INPUT(cls_targets_at_level);

  return focal_loss_forward_cuda(cls_output, cls_targets_at_level,
                                 num_positives_sum, num_classes, alpha, gamma,
                                 smoothing_factor);
}

at::Tensor focal_loss_backward(const at::Tensor &grad_output,
                               const at::Tensor &cls_output,
                               const at::Tensor &cls_targets_at_level,
                               const int64_t num_positives_sum,
                               const int64_t num_classes, const float alpha,
                               const float gamma,
                               const float smoothing_factor) {
  CHECK_INPUT(grad_output);
  CHECK_INPUT(cls_output);
  CHECK_INPUT(cls_targets_at_level);

  return focal_loss_backward_cuda(grad_output, cls_output, cls_targets_at_level,
                                  num_positives_sum, num_classes, alpha, gamma,
                                  smoothing_factor);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &focal_loss_forward,
        "Focal loss calculation forward (CUDA)");
  m.def("backward", &focal_loss_backward,
        "Focal loss calculation backward (CUDA)");
}
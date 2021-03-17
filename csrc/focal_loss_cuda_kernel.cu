#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THC.h>

template <bool SMOOTHING, typename scalar_t, typename labelscalar_t,
          typename accscalar_t, typename outscalar_t>
__global__ void focal_loss_forward_cuda_kernel(
    outscalar_t *loss, const scalar_t *__restrict__ cls_output,
    const labelscalar_t *__restrict__ cls_targets_at_level,
    const int64_t num_positives_sum, const int64_t num_examples,
    const int64_t num_classes, const float alpha, const float gamma,
    const float smoothing_factor) {
  extern __shared__ __align__(sizeof(accscalar_t)) unsigned char shm[];
  accscalar_t *loss_shm = reinterpret_cast<accscalar_t *>(shm);
  loss_shm[threadIdx.x] = 0;
  accscalar_t loss_acc = 0;

  accscalar_t one = accscalar_t(1.0);
  accscalar_t K = accscalar_t(2.0);
  accscalar_t nn_norm, np_norm, pn_norm, pp_norm;

  // *_norm is used for label smoothing only
  if (SMOOTHING) {
    nn_norm = one - smoothing_factor / K;
    np_norm = smoothing_factor / K;
    pn_norm = smoothing_factor - smoothing_factor / K;
    pp_norm = one - smoothing_factor + smoothing_factor / K;
  }

  // Accumulate loss on each thread
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < num_examples * num_classes; i += gridDim.x * blockDim.x) {
    int64_t idy = i / num_classes;
    labelscalar_t y = cls_targets_at_level[idy];

    // Skip ignored matches
    if (y == -2)
      continue;

    int64_t pos_idx = idy * num_classes + y;
    scalar_t p = cls_output[i];
    accscalar_t sigma = one / (one + std::exp(-p));
    accscalar_t loss_t;
    if (SMOOTHING) {
      accscalar_t log_sigma = ::log(sigma);
      accscalar_t log_om_sigma = ::log(one - sigma);

      // Negative matches
      loss_t = (one - alpha) * ::pow(sigma, gamma) *
               (nn_norm * log_om_sigma + np_norm * log_sigma);

      // Positive matches
      if (y >= 0 && i == pos_idx) {
        loss_t = alpha * ::pow(one - sigma, gamma) *
                 (pn_norm * log_om_sigma + pp_norm * log_sigma);
      }
    } else {
      // Negative matches
      loss_t = (one - alpha) * ::pow(sigma, gamma) * ::log(one - sigma);

      // Positive matches
      if (y >= 0 && i == pos_idx) {
        loss_t = alpha * ::pow(one - sigma, gamma) * ::log(sigma);
      }
    }

    loss_acc += loss_t;
  }
  loss_shm[threadIdx.x] = loss_acc;

  // Intra-CTA reduction
  __syncthreads();
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      loss_shm[threadIdx.x] += loss_shm[threadIdx.x + s];
    }
    __syncthreads();
  }

  // Inter-CTA reduction
  if (threadIdx.x == 0) {
    // Normalize and keep sign
    loss_acc = loss_shm[0] * -one / num_positives_sum;
    atomicAdd(loss, loss_acc);
  }
}

template <bool SMOOTHING, int ILP, typename scalar_t, typename labelscalar_t,
          typename accscalar_t, typename outscalar_t>
__global__ void focal_loss_backward_cuda_kernel(
    scalar_t *grad_input, const outscalar_t *__restrict__ grad_output,
    const scalar_t *__restrict__ cls_output,
    const labelscalar_t *__restrict__ cls_targets_at_level,
    const int64_t num_positives_sum, const int64_t num_examples,
    const int64_t num_classes, const float alpha, const float gamma,
    const float smoothing_factor) {
  int64_t idx = blockIdx.x * blockDim.x * ILP + threadIdx.x;
  accscalar_t one = accscalar_t(1.0);
  accscalar_t K = accscalar_t(2.0);
  accscalar_t grad = 0.0;
  accscalar_t normalizer =
      static_cast<accscalar_t>(grad_output[0]) / num_positives_sum;
  accscalar_t nn_norm, np_norm, pn_norm, pp_norm;

  // *_norm is used for label smoothing only
  if (SMOOTHING) {
    nn_norm = one - smoothing_factor / K;
    np_norm = smoothing_factor / K;
    pn_norm = smoothing_factor - smoothing_factor / K;
    pp_norm = one - smoothing_factor + smoothing_factor / K;
  }

#pragma unroll(ILP)
  for (int i = 0; i < ILP; i++, idx += blockDim.x) {
    if (idx >= num_examples * num_classes)
      break;

    int64_t idy = idx / num_classes;
    labelscalar_t y = cls_targets_at_level[idy];

    if (y == -2) {
      // Ignored matches
      grad = 0.0;
    } else {
      int64_t pos_idx = idy * num_classes + y;
      scalar_t p = cls_output[idx];
      accscalar_t sigma = one / (one + std::exp(-p));

      if (SMOOTHING) {
        accscalar_t log_sigma = ::log(sigma);
        accscalar_t log_om_sigma = ::log(one - sigma);

        // Negative matches
        grad = (alpha - one) * ::pow(sigma, gamma) *
               (gamma * (one - sigma) *
                    (nn_norm * log_om_sigma + np_norm * log_sigma) +
                np_norm - sigma);

        // Positive matches
        if (y >= 0 && idx == pos_idx)
          grad =
              -alpha * ::pow(one - sigma, gamma) *
              (-gamma * sigma * (pn_norm * log_om_sigma + nn_norm * log_sigma) +
               nn_norm - sigma);
      } else {
        // Negative matches
        grad = (alpha - one) * ::pow(sigma, gamma) *
               (gamma * (one - sigma) * ::log(one - sigma) - sigma);

        // Positive matches
        if (y >= 0 && idx == pos_idx)
          grad = -alpha * ::pow(one - sigma, gamma) *
                 (one - sigma - gamma * sigma * ::log(sigma));
      }
    }

    grad_input[idx] = grad * normalizer;
  }
}

at::Tensor focal_loss_forward_cuda(const at::Tensor &cls_output,
                                   const at::Tensor &cls_targets_at_level,
                                   const int64_t num_positives_sum,
                                   const int64_t num_classes, const float alpha,
                                   const float gamma,
                                   const float smoothing_factor) {
  AT_ASSERTM(cls_output.numel() % num_classes == 0, "Invalid input shape.");
  AT_ASSERTM(cls_targets_at_level.scalar_type() == at::kLong,
             "Invalid label type.");
  AT_ASSERTM(num_positives_sum > 0,
             "Expect more than one positive matches, user should increase "
             "total matches by one.");
  AT_ASSERTM(num_classes > 0, "Expect more than one classes.");

  int64_t num_examples = cls_output.numel() / num_classes;
  at::Tensor loss = at::zeros({}, cls_output.options().dtype(at::kFloat));

  // The grid contains 2 CTA per SM, each CTA loop on input with stride till the
  // last item.
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, at::cuda::current_device());
  dim3 block(512);
  dim3 grid(2 * props.multiProcessorCount);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (smoothing_factor == 0.0f) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        cls_output.scalar_type(), "focal_loss_fprop", [&] {
          using accscalar_t = at::acc_type<scalar_t, true>;
          using labelscalar_t = int64_t;
          using outscalar_t = float;
          focal_loss_forward_cuda_kernel<false, scalar_t, labelscalar_t,
                                         accscalar_t, outscalar_t>
              <<<grid, block, block.x * sizeof(accscalar_t), stream>>>(
                  loss.data_ptr<outscalar_t>(), cls_output.data_ptr<scalar_t>(),
                  cls_targets_at_level.data_ptr<labelscalar_t>(),
                  num_positives_sum, num_examples, num_classes, alpha, gamma,
                  smoothing_factor);
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        cls_output.scalar_type(), "focal_loss_fprop", [&] {
          using accscalar_t = at::acc_type<scalar_t, true>;
          using labelscalar_t = int64_t;
          using outscalar_t = float;
          focal_loss_forward_cuda_kernel<true, scalar_t, labelscalar_t,
                                         accscalar_t, outscalar_t>
              <<<grid, block, block.x * sizeof(accscalar_t), stream>>>(
                  loss.data_ptr<outscalar_t>(), cls_output.data_ptr<scalar_t>(),
                  cls_targets_at_level.data_ptr<labelscalar_t>(),
                  num_positives_sum, num_examples, num_classes, alpha, gamma,
                  smoothing_factor);
        });
  }

  THCudaCheck(cudaGetLastError());
  return loss;
}

at::Tensor focal_loss_backward_cuda(const at::Tensor &grad_output,
                                    const at::Tensor &cls_output,
                                    const at::Tensor &cls_targets_at_level,
                                    const int64_t num_positives_sum,
                                    const int64_t num_classes,
                                    const float alpha, const float gamma,
                                    const float smoothing_factor) {
  int64_t num_examples = cls_output.numel() / num_classes;
  at::Tensor grad_input = at::empty_like(cls_output);

  // Each thread process ILP elements
  const int ILP = 8;
  dim3 block(512);
  dim3 grid((cls_output.numel() + block.x * ILP - 1) / (block.x * ILP));

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (smoothing_factor == 0.0f) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        cls_output.scalar_type(), "focal_loss_bprop", [&] {
          using accscalar_t = at::acc_type<scalar_t, true>;
          using labelscalar_t = int64_t;
          using outscalar_t = float;
          focal_loss_backward_cuda_kernel<false, ILP, scalar_t, labelscalar_t,
                                          accscalar_t, outscalar_t>
              <<<grid, block, 0, stream>>>(
                  grad_input.data_ptr<scalar_t>(),
                  grad_output.data_ptr<outscalar_t>(),
                  cls_output.data_ptr<scalar_t>(),
                  cls_targets_at_level.data_ptr<labelscalar_t>(),
                  num_positives_sum, num_examples, num_classes, alpha, gamma,
                  smoothing_factor);
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        cls_output.scalar_type(), "focal_loss_bprop", [&] {
          using accscalar_t = at::acc_type<scalar_t, true>;
          using labelscalar_t = int64_t;
          using outscalar_t = float;
          focal_loss_backward_cuda_kernel<true, ILP, scalar_t, labelscalar_t,
                                          accscalar_t, outscalar_t>
              <<<grid, block, 0, stream>>>(
                  grad_input.data_ptr<scalar_t>(),
                  grad_output.data_ptr<outscalar_t>(),
                  cls_output.data_ptr<scalar_t>(),
                  cls_targets_at_level.data_ptr<labelscalar_t>(),
                  num_positives_sum, num_examples, num_classes, alpha, gamma,
                  smoothing_factor);
        });
  }

  THCudaCheck(cudaGetLastError());
  return grad_input;
}

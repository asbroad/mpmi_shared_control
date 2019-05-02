#include <iostream>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <math.h>

int copy_data_to_device_float(cublasStatus_t status, const int m, const int n, float* host_data, float* device_data) {
  status = cublasSetMatrix(m, n, sizeof(float), host_data, m, device_data, m);
  if (status != CUBLAS_STATUS_SUCCESS) {
     std::cerr << "***cublasSetMatrix failed***\n";
     return 2;
  }
  return 0;
}

int copy_data_to_device_int(cublasStatus_t status, const int m, const int n, int* host_data, int* device_data){
  status = cublasSetMatrix(m, n, sizeof(int), host_data, m, device_data, m);
  if (status != CUBLAS_STATUS_SUCCESS) {
     std::cerr << "***cublasSetMatrix failed***\n";
     return 2;
  }
  return 0;
}

int copy_data_to_host_float(cublasStatus_t status, const int m, const int n, float* host_data, float* device_data){
  status = cublasGetMatrix(m, n, sizeof(float), device_data, m, host_data, m);
  if (status != CUBLAS_STATUS_SUCCESS) {
     std::cerr << "***cublasGetMatrix failed***\n";
     return 2;
  }
  return 0;
}

int copy_data_to_host_int(cublasStatus_t status, const int m, const int n, int* host_data, int* device_data){
  status = cublasGetMatrix(m, n, sizeof(int), device_data, m, host_data, m);
  if (status != CUBLAS_STATUS_SUCCESS) {
     std::cerr << "***cublasGetMatrix failed***\n";
     return 2;
  }
  return 0;
}

int forward_predict(cublasStatus_t status, cublasHandle_t handle, const int m, const int n, const int k, const float alpha, const float beta, float* device_A, float* device_B) {

  status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, device_A, m, device_B, m, &beta, device_B, m);
  if (status != CUBLAS_STATUS_SUCCESS) {
     std::cerr << "***cublasSgemm forward prediction failed***\n";
     return 2;
  }

  return 0;
}

__global__ void generate_samples_gpu_helper(const int num_samples, const int dim_koopman, const float* device_control, float* device_states) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int sample_start_idx;

  for (int idx = index; idx < num_samples; idx+=stride) {

    sample_start_idx = idx * dim_koopman;
    device_states[sample_start_idx] = 1.0;
    // device_states[sample_start_idx + 1] // the original state space remains the same, i.e. this is a predicted value
    // device_states[sample_start_idx + 2] //
    // device_states[sample_start_idx + 3] //
    device_states[sample_start_idx + 4]  = device_control[idx]; // reset the control value, can add noise here
    device_states[sample_start_idx + 5]  = device_states[sample_start_idx + 3]*cos(device_states[sample_start_idx + 1]);
  }

}

int generate_samples_gpu(const int block_size, const int num_blocks, const int num_samples, const int dim_koopman, const float* device_control, float* device_states) {
  cudaError_t last_error;
  generate_samples_gpu_helper<<<num_blocks, block_size>>>(num_samples, dim_koopman, device_control, device_states);
  last_error = cudaGetLastError();
  cudaDeviceSynchronize();
  if (last_error == cudaErrorInvalidConfiguration) {
    std::cerr << "***Invalid CUDA configuration (e.g. too many threads)" << std::endl;
    return 2;
  } else if (last_error != cudaSuccess) {
    std::cerr << "***Re-projection into Koopman space failed***\n";
    return 2;
  }
  return 0;
}

__global__ void update_safe_metric_helper(const int num_samples, const float inflated_max_angle, const int dim_koopman, float* device_states, int* device_safe) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int sample_start_idx;

  for (int idx = index; idx < num_samples; idx+=stride) {

    if (device_safe[idx] == true) {
      sample_start_idx = idx * dim_koopman;
      if (device_states[sample_start_idx + 1] > inflated_max_angle || device_states[sample_start_idx + 1] < -inflated_max_angle) {
        device_safe[idx] = false;
      }
    }
  }

}

int update_safe_metric(const int block_size, const int num_blocks, const int num_samples, const float inflated_max_angle, const int dim_koopman, float* device_states, int* device_safe) {
  cudaError_t last_error;
  update_safe_metric_helper<<<num_blocks, block_size>>>(num_samples, inflated_max_angle, dim_koopman, device_states, device_safe);
  last_error = cudaGetLastError();
  cudaDeviceSynchronize();
  if (last_error == cudaErrorInvalidConfiguration) {
    std::cerr << "***Invalid CUDA configuration (e.g. too many threads)" << std::endl;
    return 2;
  } else if (last_error != cudaSuccess) {
    std::cerr << "***Re-projection into Koopman space failed***\n";
    return 2;
  }
  return 0;
}


__global__ void compute_optimality_metric_helper(const float user_control, const float* device_control, const int num_samples, const int* device_safe, float* device_optimality_metric) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int idx = index; idx < num_samples; idx+=stride) {
    if (device_safe[idx] == 1) { // if it's safe
      device_optimality_metric[idx] = sqrt(pow((device_control[idx] - user_control), 2));
    } else {
      device_optimality_metric[idx] = 100000000000000;
    }
  }

}

int compute_optimality_metric(const int block_size, const int num_blocks, const float user_control, const float* device_control, const int num_samples, const int* device_safe, float* device_optimality_metric) {
  cudaError_t last_error;
  compute_optimality_metric_helper<<<num_blocks, block_size>>>(user_control, device_control, num_samples, device_safe, device_optimality_metric);
  last_error = cudaGetLastError();
  cudaDeviceSynchronize();
  if (last_error == cudaErrorInvalidConfiguration) {
    std::cerr << "***Invalid CUDA configuration (e.g. too many threads)" << std::endl;
    return 2;
  } else if (last_error != cudaSuccess) {
    std::cerr << "***Compute optimality metric failed***\n";
    return 2;
  }
  return 0;
}

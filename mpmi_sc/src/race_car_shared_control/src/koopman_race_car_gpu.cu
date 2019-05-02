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

// http://geomalgorithms.com/code.html
__global__ void in_poly_gpu_helper(const int poly_idx, const int point_size, const int poly_size, const int points_defining_poly, const float* polys, const float* points, const int dim_koopman, const int num_samples, int* host_point_in_poly_gpu) {

  // polygon is not defined in koopman space
  int poly_idx_d = poly_idx * point_size * points_defining_poly; // start of polygon to check points against, poly is not defined in koopman space
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int idx = index; idx < num_samples; idx+=stride) {

      float x = points[dim_koopman*idx + 1];
      float y = points[dim_koopman*idx + 2];

      if (host_point_in_poly_gpu[idx] == 0) {

        int    cn = 0;    // the  crossing number counter
        for (int i = 0; i < poly_size; i++) {    // edge from V[i]  to V[i+1]
           if (((polys[poly_idx_d + point_size*i + 1] <= y) && (polys[poly_idx_d + point_size*(i+1) + 1] > y)) // an upward crossing
            || ((polys[poly_idx_d + point_size*i + 1] > y) && (polys[poly_idx_d + point_size*(i+1) + 1] <= y))) { // a downward crossing
                // compute  the actual edge-ray intersect x-coordinate
                float vt = (float)(y - polys[poly_idx_d + point_size*i + 1]) / (polys[poly_idx_d + point_size*(i+1) + 1] - polys[poly_idx_d + point_size*i + 1]);
                if (x < polys[poly_idx_d + point_size*i] + vt * (polys[poly_idx_d + point_size*(i+1)] - polys[poly_idx_d + point_size*i])) // P.x < intersect
                     ++cn;   // a valid crossing of y=P.y right of P.x
            }
        }

        int ret = cn&1;
        if (ret == 1) {
          host_point_in_poly_gpu[idx] = 1;
        }
      }

  }

}

int in_poly_gpu(const int block_size, const int num_blocks, const int poly_idx, const int point_size, const int poly_size, const int points_defining_poly, const float* polys, const float* points, const int dim_koopman, const int num_samples, int* host_point_in_poly_gpu) {

  cudaError_t last_error;
  in_poly_gpu_helper<<<num_blocks, block_size>>>(poly_idx, point_size, poly_size, points_defining_poly, polys, points, dim_koopman, num_samples, host_point_in_poly_gpu);
  last_error = cudaGetLastError();
  cudaDeviceSynchronize();

  if (last_error == cudaErrorInvalidConfiguration) {
    std::cerr << "***Invalid CUDA configuration (e.g. too many threads)" << std::endl;
    return 2;
  } else if (last_error != cudaSuccess) {
    std::cerr << "***Safety check failed***\n";
    return 2;
  }
  return 0;

}

__global__ void generate_samples_gpu_helper(const int num_samples, const int dim_koopman, const float* device_control_heading, const float* device_control_gas, const float* device_control_break, float* device_states) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int sample_start_idx;

  for (int idx = index; idx < num_samples; idx+=stride) {

    sample_start_idx = idx * dim_koopman;
    device_states[sample_start_idx] = 1.0;
    // device_states[sample_start_idx + 1] // the original state space remains the same, i.e. this is a predicted value
    // device_states[sample_start_idx + 2] //
    // device_states[sample_start_idx + 3] //
    // device_states[sample_start_idx + 4] //
    // device_states[sample_start_idx + 5] //
    // device_states[sample_start_idx + 6] //
    device_states[sample_start_idx + 7]  = device_control_heading[idx]; // reset the control value, can add noise here
    device_states[sample_start_idx + 8]  = device_control_gas[idx];
    device_states[sample_start_idx + 9]  = device_control_break[idx];
    device_states[sample_start_idx + 10] = device_states[sample_start_idx + 7]*device_states[sample_start_idx + 1]; // control * state
    device_states[sample_start_idx + 11] = device_states[sample_start_idx + 7]*device_states[sample_start_idx + 2];
    device_states[sample_start_idx + 12] = device_states[sample_start_idx + 7]*device_states[sample_start_idx + 4];
    device_states[sample_start_idx + 13] = device_states[sample_start_idx + 7]*device_states[sample_start_idx + 5];
    device_states[sample_start_idx + 14] = device_states[sample_start_idx + 8]*device_states[sample_start_idx + 1];
    device_states[sample_start_idx + 15] = device_states[sample_start_idx + 8]*device_states[sample_start_idx + 2];
    device_states[sample_start_idx + 16] = device_states[sample_start_idx + 8]*device_states[sample_start_idx + 4];
    device_states[sample_start_idx + 17] = device_states[sample_start_idx + 8]*device_states[sample_start_idx + 5];
    device_states[sample_start_idx + 18] = device_states[sample_start_idx + 9]*device_states[sample_start_idx + 1];
    device_states[sample_start_idx + 19] = device_states[sample_start_idx + 9]*device_states[sample_start_idx + 4];
    device_states[sample_start_idx + 20] = device_states[sample_start_idx + 9]*device_states[sample_start_idx + 5];
    device_states[sample_start_idx + 21] = device_states[sample_start_idx + 1]*cos(device_states[sample_start_idx + 3]); // heading * state
    device_states[sample_start_idx + 22] = device_states[sample_start_idx + 1]*sin(device_states[sample_start_idx + 3]);
    device_states[sample_start_idx + 23] = device_states[sample_start_idx + 2]*sin(device_states[sample_start_idx + 3]);
    device_states[sample_start_idx + 24] = device_states[sample_start_idx + 4]*cos(device_states[sample_start_idx + 3]);
    device_states[sample_start_idx + 25] = device_states[sample_start_idx + 5]*cos(device_states[sample_start_idx + 3]);
    device_states[sample_start_idx + 26] = device_states[sample_start_idx + 4]*device_states[sample_start_idx + 1]; // velocity * position
    device_states[sample_start_idx + 27] = device_states[sample_start_idx + 5]*device_states[sample_start_idx + 2];
    device_states[sample_start_idx + 28] = device_states[sample_start_idx + 6]*device_states[sample_start_idx + 3];
    device_states[sample_start_idx + 29] = device_states[sample_start_idx + 1]*cos(device_states[sample_start_idx + 6]); // heading velocity * state
    device_states[sample_start_idx + 30] = device_states[sample_start_idx + 1]*sin(device_states[sample_start_idx + 6]);
    device_states[sample_start_idx + 31] = device_states[sample_start_idx + 2]*cos(device_states[sample_start_idx + 6]);
    device_states[sample_start_idx + 32] = device_states[sample_start_idx + 2]*sin(device_states[sample_start_idx + 6]);
    device_states[sample_start_idx + 33] = device_states[sample_start_idx + 4]*cos(device_states[sample_start_idx + 6]);
    device_states[sample_start_idx + 34] = device_states[sample_start_idx + 4]*sin(device_states[sample_start_idx + 6]);
    device_states[sample_start_idx + 35] = device_states[sample_start_idx + 5]*sin(device_states[sample_start_idx + 6]);
  }

}

int generate_samples_gpu(const int block_size, const int num_blocks, const int num_samples, const int dim_koopman, const float* device_control_heading, const float* device_control_gas, const float* device_control_break, float* device_states) {
  cudaError_t last_error;
  generate_samples_gpu_helper<<<num_blocks, block_size>>>(num_samples, dim_koopman, device_control_heading, device_control_gas, device_control_break, device_states);
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

__global__ void update_safe_metric_helper(const int num_samples, int* device_point_in_poly, int* device_safe) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int idx = index; idx < num_samples; idx+=stride) {
    if (device_point_in_poly[idx] == 0) {
      device_safe[idx] = 0;
    }
  }

}

int update_safe_metric(const int block_size, const int num_blocks, const int num_samples, int* device_point_in_poly, int* device_safe) {
  cudaError_t last_error;
  update_safe_metric_helper<<<num_blocks, block_size>>>(num_samples, device_point_in_poly, device_safe);
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


__global__ void compute_optimality_metric_helper(const float user_control_heading, const float user_control_gas, const float user_control_break, const float* device_control_heading, const float* device_control_gas, const float* device_control_break, const int num_samples, const int* device_safe, float* device_optimality_metric) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int idx = index; idx < num_samples; idx+=stride) {
    if (device_safe[idx] == 1) { // if it's safe
      device_optimality_metric[idx] = sqrt(pow((device_control_heading[idx] - user_control_heading), 2) + pow((device_control_gas[idx] - user_control_gas), 4) + pow((device_control_break[idx] - user_control_break), 4));
    } else {
      device_optimality_metric[idx] = 100000000000000;
    }
  }

}

int compute_optimality_metric(const int block_size, const int num_blocks, const float user_control_heading, const float user_control_gas, const float user_control_break, const float* device_control_heading, const float* device_control_gas, const float* device_control_break, const int num_samples, const int* device_safe, float* device_optimality_metric) {
  cudaError_t last_error;
  compute_optimality_metric_helper<<<num_blocks, block_size>>>(user_control_heading, user_control_gas, user_control_break, device_control_heading, device_control_gas, device_control_break, num_samples, device_safe, device_optimality_metric);
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

__global__ void compute_current_road_poly_helper(const float x, const float y, const int num_polys, const int point_size, const int points_defining_poly, const int poly_size, const float* polys, int* device_current_poly) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int idx = index; idx < num_polys; idx+=stride) {

    int poly_idx_d = idx * point_size * points_defining_poly; // start of polygon to check points against, poly is not defined in koopman space

    int    cn = 0;    // the  crossing number counter
    for (int i = 0; i < poly_size; i++) {    // edge from V[i]  to V[i+1]
       if (((polys[poly_idx_d + point_size*i + 1] <= y) && (polys[poly_idx_d + point_size*(i+1) + 1] > y)) // an upward crossing
        || ((polys[poly_idx_d + point_size*i + 1] > y) && (polys[poly_idx_d + point_size*(i+1) + 1] <= y))) { // a downward crossing
            // compute  the actual edge-ray intersect x-coordinate
            float vt = (float)(y - polys[poly_idx_d + point_size*i + 1]) / (polys[poly_idx_d + point_size*(i+1) + 1] - polys[poly_idx_d + point_size*i + 1]);
            if (x < polys[poly_idx_d + point_size*i] + vt * (polys[poly_idx_d + point_size*(i+1)] - polys[poly_idx_d + point_size*i])) // P.x < intersect
                 ++cn;   // a valid crossing of y=P.y right of P.x
        }
    }
    device_current_poly[idx] = cn&1;
  }

}

int compute_current_road_poly(const int block_size, const int num_blocks, const float x, const float y, const int num_polys, const int point_size, const int points_defining_poly, const int poly_size, const float* device_road_poly, int* device_current_poly) {
  cudaError_t last_error;
  compute_current_road_poly_helper<<<num_blocks, block_size>>>(x, y, num_polys, point_size, points_defining_poly, poly_size, device_road_poly, device_current_poly);
  last_error = cudaGetLastError();
  cudaDeviceSynchronize();
  if (last_error == cudaErrorInvalidConfiguration) {
    std::cerr << "***Invalid CUDA configuration (e.g. too many threads)" << std::endl;
    return 2;
  } else if (last_error != cudaSuccess) {
    std::cerr << "***Compute current road poly failed***\n";
    return 2;
  }
  return 0;
}

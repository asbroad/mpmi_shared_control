#include <ctime>
#include <stdlib.h>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <math.h>
#include <algorithm>
// ROS
#include <ros/ros.h>
#include <signal.h>
#include <std_msgs/String.h>
#include <std_msgs/Float32.h>
#include <balance_bot_shared_control/Mpmi.h>
// Modeling
#include "model/koopman_balance_bot.hpp"
#include "model/basis_balance_bot.hpp"
// Cuda
#include <cuda_runtime.h>
#include "cublas_v2.h"

// Function definitions for code run on the GPU
int copy_data_to_device_float(cublasStatus_t status, const int m, const int n, float* host_data, float* device_data);
int copy_data_to_device_int(cublasStatus_t status, const int m, const int n, int* host_data, int* device_data);
int copy_data_to_host_float(cublasStatus_t status, const int m, const int n, float* host_data, float* device_data);
int copy_data_to_host_int(cublasStatus_t status, const int m, const int n, int* host_data, int* device_data);

int generate_samples_gpu(const int block_size, const int num_blocks, const int num_samples, const int dim_koopman, const float* device_control, float* device_states);
int forward_predict(cublasStatus_t status, cublasHandle_t handle, const int m, const int n, const int k, const float alpha, const float beta, float* device_A, float* device_B);
int update_safe_metric(const int block_size, const int num_blocks, const int num_samples, const float inflated_max_angle, const int dim_koopman, float* device_states, int* device_safe);
int compute_optimality_metric(const int block_size, const int num_blocks, const float user_control, const float* device_control, const int num_samples, const int* device_safe, float* device_optimality_metric);

class ModelPredictiveControl {

  public:

    // parameters
    std::string modelFilepath;
    int dim_state = 3;
    int dim_control = 1;
    int dim_state_and_control = dim_state + dim_control;
    int dim_koopman = 6;
    float max_angle = 1.768f;
    float inflation;
    float inflated_max_angle;
    int num_samples = 10000;
    int time_horizon;
    int optimal_index = 0;
    int block_size = 512; // must be set for each GPU, there are some checks later that will tell you if you did this incorrectly
    int num_blocks = (num_samples + block_size - 1) / block_size;

    // data vectors
    float* mpmi_control;

    // set up host variables
    float* host_koopman;
    float* host_states;
    float* host_cur_state;
    float* host_cur_state_projected;
    int* host_safe;
    float* host_optimality_metric;

    // set up device variables
    float* device_koopman;
    float* device_states;
    int* device_safe;
    float* device_optimality_metric;

    // define cublas parameters
    cublasStatus_t status;
    cublasHandle_t handle;
    float alpha = 1.0f;
    float beta  = 0.0f;
    int lda = dim_koopman;

    // equally spaced control samples (inputs)
    float* host_control;
    float* device_control;

    // variables for optimal solution
    float best_optimality_measure;
    float cur_optimality_measure;
    bool safe_optimal_solution;
    float percent_safe;

    // set up ros services, subscribers and publishers
    ros::ServiceServer service;
    ros::Subscriber inflation_sub;

    // Koopman model
    KoopmanOperator* koopman_operator;

    ModelPredictiveControl(ros::Rate* loop_rate) {

      // set up node
      ros::NodeHandle nh;

      nh.getParam("/time_horizon", time_horizon);
      nh.getParam("/inflation_radius", inflation);
      inflated_max_angle = max_angle - inflation;

      // set up subscribers
      service = nh.advertiseService("/mpmi_control", &ModelPredictiveControl::mpmi_cb, this);
      inflation_sub = nh.subscribe("/inflation_update", 1, &ModelPredictiveControl::inflation_cb, this);

      // set up host variables
      host_cur_state = new float[dim_state_and_control]();
      host_cur_state_projected = new float[dim_koopman]();
      host_koopman = new float[dim_koopman*dim_koopman]();
      host_states = new float[dim_koopman*num_samples]();
      host_safe = new int[num_samples]();
      host_optimality_metric = new float[num_samples]();
      host_control = linspace(-1.0, 1.0, num_samples);       // generate control samples

      // set up device variables
      cudaMalloc((void**)&device_koopman, dim_koopman * dim_koopman * sizeof(float));
      cudaMalloc((void**)&device_states, dim_koopman * num_samples * sizeof(float));
      cudaMalloc((void**)&device_safe, num_samples * sizeof(int));
      cudaMalloc((void**)&device_control, num_samples * sizeof(float));
      cudaMalloc((void**)&device_optimality_metric, num_samples * sizeof(float));

      // set up cublas
      status = cublasCreate(&handle);
      if (status != CUBLAS_STATUS_SUCCESS) {
         std::cerr << "***cublasCreate failed***\n";
      }

      // initialize safety array and send to device
      std::fill(host_safe, host_safe+num_samples, 1);
      copy_data_to_device_int(status, num_samples, 1, host_safe, device_safe);

      // load Koopman operator and send to gpu
      nh.getParam("/model_path", modelFilepath);
      std::string nonlinearFilePath = modelFilepath;
      std::cout << "Model path : " << modelFilepath << std::endl;
      koopman_operator = new KoopmanOperator(new Basis());
      koopman_operator->loadOperator(nonlinearFilePath);
      copy_data_to_device_float(status, dim_koopman, dim_koopman, koopman_operator->_K_gpu, device_koopman);

      // copy control samples to GPU
      copy_data_to_device_float(status, num_samples, 1, host_control, device_control);

      std::cout << "Using " << num_samples << " control samples." << std::endl;
      std::cout << "Using " << num_blocks << " CUDA block(s)." << std::endl;

    }

    bool mpmi_cb(balance_bot_shared_control::Mpmi::Request &req, balance_bot_shared_control::Mpmi::Response &res) {

      // get state
      host_cur_state[0] = req.angle;
      host_cur_state[1] = req.angle_dot;
      host_cur_state[2] = req.velocity;
      host_cur_state[3] = req.u_1;

      // set up projected states array (in Koopman space)
      int tmp_counter = 0;
      for (int i = 0; i < num_samples*dim_koopman; i+=dim_koopman) {
        // get sampled control and add to state
        host_cur_state[3] = host_control[tmp_counter];
        // project using basis
        host_cur_state_projected = koopman_operator->basis->project(host_cur_state);
        // add to states array
        std::copy(host_cur_state_projected, host_cur_state_projected + dim_koopman, host_states + i);
        tmp_counter += 1;
      }

      // copy states to gpu
      copy_data_to_device_float(status, dim_koopman, num_samples, host_states, device_states);

      // re-initialize safety array
      std::fill(host_safe, host_safe+num_samples, 1);
      copy_data_to_device_int(status, num_samples, 1, host_safe, device_safe);

      // perform forward prediction and safety check on GPU
      for (int i = 0; i < time_horizon; i++) {

        // forward predict
        forward_predict(status, handle, dim_koopman, num_samples, dim_koopman, alpha, beta, device_koopman, device_states);

        // update koopman space state representation
        generate_samples_gpu(block_size, num_blocks, num_samples, dim_koopman, device_control, device_states);

        // perform safety check
        update_safe_metric(block_size, num_blocks, num_samples, inflated_max_angle, dim_koopman, device_states, device_safe);

      }

      // copy safety array back
      copy_data_to_host_int(status, num_samples, 1, host_safe, device_safe);

      // compute optimality of control inputs
      compute_optimality_metric(block_size, num_blocks, req.u_1, device_control, num_samples, device_safe, device_optimality_metric);
      copy_data_to_host_float(status, num_samples, 1, host_optimality_metric, device_optimality_metric);

      // sort control inputs by optimality
      std::vector<float> optimality_metric_std (host_optimality_metric, host_optimality_metric + num_samples);
      std::vector<size_t> sorted_optimality_metric_indices = sort_indexes(optimality_metric_std);
      best_optimality_measure = host_optimality_metric[sorted_optimality_metric_indices[0]];

      // compute percent safe
      int num_safe = 0;
      for (int i = 0; i < num_samples; i++) {
        if (host_safe[i] == 1) {
          num_safe += 1.0;
        }
      }
      percent_safe = num_safe/float(num_samples);

      // // set return value
      int pick_opt_solution = 0;
      if (num_samples < pick_opt_solution) {
        pick_opt_solution = num_samples;
      }
      if (percent_safe < 0.0000001) { // if we can't find any safe options, just let the user do what they want (system failure)
        res.u_1 = req.u_1;
        res.percent_safe = 0.0;
        res.cost = 1000000.0;
      } else {
        res.u_1 = host_control[sorted_optimality_metric_indices[optimal_index]];
        res.percent_safe = percent_safe;
        res.cost = best_optimality_measure;
      }

      return true;
    }

    void inflation_cb(const std_msgs::Float32::ConstPtr& msg) {
      inflation = msg->data;
      inflated_max_angle = max_angle - inflation;
    }

    float* linspace(float a, float b, int n) {
        float* array = new float[n]();
        float step = (b-a) / (n-1);
        for (int i = 0; i < n; i++) {
          array[i] = std::max(std::min(a, 1.0f), -1.0f);
          a += step;
        }
        return array;
    }

    template <typename T>
    std::vector<size_t> sort_indexes(const std::vector<T> &v) {

      // initialize original index locations
      std::vector<size_t> idx(v.size());
      iota(idx.begin(), idx.end(), 0);

      // sort indexes based on comparing values in v
      sort(idx.begin(), idx.end(),
           [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

      return idx;
    }

};


int main(int argc, char** argv) {

  ros::init(argc, argv,"shared_control_balance_bot");
  ros::NodeHandle nh;

  ros::Rate loop_rate(100);
  ModelPredictiveControl sys(&loop_rate);

  while (ros::ok()) {
    loop_rate.sleep();
    ros::spinOnce();
  }

  return 0;
}

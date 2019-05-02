#include <ctime>
#include <stdlib.h>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <math.h>
// ROS
#include <ros/ros.h>
#include <signal.h>
#include <std_msgs/String.h>
#include <race_car_shared_control/RoadPoly.h>
#include <race_car_shared_control/Mpmi.h>
// Modeling
#include "model/koopman_race_car.hpp"
#include "model/basis_race_car.hpp"
// Cuda
#include <cuda_runtime.h>
#include "cublas_v2.h"
// point class
#include "point.h"

// Function definitions for code run on the GPU
int copy_data_to_device_float(cublasStatus_t status, const int m, const int n, float* host_data, float* device_data);
int copy_data_to_device_int(cublasStatus_t status, const int m, const int n, int* host_data, int* device_data);
int copy_data_to_host_float(cublasStatus_t status, const int m, const int n, float* host_data, float* device_data);
int copy_data_to_host_int(cublasStatus_t status, const int m, const int n, int* host_data, int* device_data);

int generate_samples_gpu(const int block_size, const int num_blocks, const int num_samples, const int dim_koopman, const float* device_control_heading, const float* device_control_gas, const float* device_control_break, float* device_states);
int forward_predict(cublasStatus_t status, cublasHandle_t handle, const int m, const int n, const int k, const float alpha, const float beta, float* device_A, float* device_B);
int in_poly_gpu(const int block_size, const int num_blocks, const int poly_idx, const int point_size, const int poly_size, const int points_defining_poly, const float* polys, const float* points, const int dim_koopman, const int num_samples, int* host_point_in_poly);
int update_safe_metric(const int block_size, const int num_blocks, const int num_samples, int* device_point_in_poly, int* device_safe);
int compute_optimality_metric(const int block_size, const int num_blocks, const float user_control_heading, const float user_control_gas, const float user_control_break, const float* device_control_heading, const float* device_control_gas, const float* device_control_break, const int num_samples, const int* device_safe, float* device_optimality_metric);
int compute_current_road_poly(const int block_size, const int num_blocks, const float x, const float y, const int num_polys, const int point_size, const int points_defining_poly, const int poly_size, const float* device_road_poly, int* device_current_poly);

class ModelPredictiveControl {

  public:

    // parameters
    std::string modelFilepath;
    int point_size = 2;
    int dim_state = 6;
    int dim_control = 3;
    int dim_state_and_control = dim_state + dim_control;
    int dim_koopman = 36;
    int num_samples = 10120;
    int num_samples_heading = 115;
    int num_samples_gas = 59;
    int num_samples_break = 29;
    int optimal_index = 0;
    int block_size = 512; // must be set for each GPU, there are some checks later that will tell you if you did this incorrectly
    int num_blocks = (num_samples + block_size - 1) / block_size;
    int time_horizon;;

    // data vectors
    float* mpmi_control;

    // set up host variables
    float* host_koopman;
    float* host_states;
    float* host_cur_state;
    float* host_cur_state_projected;
    float* host_road_poly;
    int* host_point_in_poly;
    int* host_safe;
    float* host_optimality_metric;
    int* host_current_poly;

    // set up device variables
    float* device_koopman;
    float* device_states;
    float* device_road_poly;
    int* device_point_in_poly;
    int* device_safe;
    float* device_optimality_metric;
    int* device_current_poly;

    // define cublas parameters
    cublasStatus_t status;
    cublasHandle_t handle;
    float alpha = 1.0f;
    float beta  = 0.0f;
    int lda = dim_koopman; // TODO: is this right?

    // equally spaced control samples (inputs)
    float* host_control_heading;
    float* host_control_gas;
    float* host_control_break;
    float* device_control_heading;
    float* device_control_gas;
    float* device_control_break;

    // set up environment variables
    int poly_size = 4;
    int num_points_in_poly = 5;
    int init_road_poly_idx = 1;
    int road_poly_idx = init_road_poly_idx;  // we always start in road poly idx 1
    int num_road_poly_check = 20; // how many polygons do we want to check if the projected state is in?
    std::vector<std::vector<Point>> road_poly;

    // variables for optimal solution
    float best_optimality_measure;
    float cur_optimality_measure;
    bool safe_optimal_solution;
    int optimal_idx;
    float percent_safe;

    // set up ros services, subscribers and publishers
    ros::Subscriber reset_sub;
    ros::ServiceServer service;
    ros::Subscriber road_poly_sub;

    // Koopman models
    KoopmanOperator* koopman_operator;

    ModelPredictiveControl(ros::Rate* loop_rate) {

      // set up node
      ros::NodeHandle nh;
      nh.getParam("/time_horizon", time_horizon);

      // set up subscribers
      reset_sub = nh.subscribe("/reset", 1, &ModelPredictiveControl::reset_cb, this);
      service = nh.advertiseService("/mpmi_control", &ModelPredictiveControl::mpmi_cb, this);
      road_poly_sub = nh.subscribe("/road_poly", 1, &ModelPredictiveControl::road_poly_cb, this);

      // set up host variables
      host_cur_state = new float[dim_state_and_control]();
      host_cur_state_projected = new float[dim_koopman]();
      host_koopman = new float[dim_koopman*dim_koopman]();
      host_states = new float[dim_koopman*num_samples]();
      host_safe = new int[num_samples]();
      host_control_heading = new float[num_samples]();
      host_control_gas = new float[num_samples]();
      host_control_break = new float[num_samples]();
      host_point_in_poly = new int[num_samples]();
      host_optimality_metric = new float[num_samples]();

      // set up device variables
      cudaMalloc((void**)&device_koopman, dim_koopman * dim_koopman * sizeof(float));
      cudaMalloc((void**)&device_states, dim_koopman * num_samples * sizeof(float));
      cudaMalloc((void**)&device_safe, num_samples * sizeof(int));
      cudaMalloc((void**)&device_control_heading, num_samples * sizeof(float));
      cudaMalloc((void**)&device_control_gas, num_samples * sizeof(float));
      cudaMalloc((void**)&device_control_break, num_samples * sizeof(float));
      cudaMalloc((void**)&device_point_in_poly, num_samples * sizeof(int));
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
      std::cout << "Model path : " << modelFilepath << std::endl;
      std::string nonlinearFilePath = modelFilepath;
      koopman_operator = new KoopmanOperator(new Basis());
      koopman_operator->loadOperator(nonlinearFilePath);
      copy_data_to_device_float(status, dim_koopman, dim_koopman, koopman_operator->_K_gpu, device_koopman);

      // generate control samples
      float* control_heading_grid = linspace(-1.0, 1.0, num_samples_heading);
      float* control_gas_grid = linspace(0.0, 1.0, num_samples_gas);
      float* control_break_grid = linspace(0.0, 0.5, num_samples_break);

      int sample_count = 0;
      for (int i = 0; i < num_samples_heading; i++) {
        for (int j = 0; j < num_samples_gas; j++) {
          host_control_heading[sample_count] = control_heading_grid[i];
          host_control_gas[sample_count] = control_gas_grid[j];
          host_control_break[sample_count] = 0.0;
          ++sample_count;
        }
      }
      for (int i = 0; i < num_samples_heading; i++) {
        for (int j = 0; j < num_samples_break; j++) {
          host_control_heading[sample_count] = control_heading_grid[i];
          host_control_gas[sample_count] = 0.0;
          host_control_break[sample_count] = control_break_grid[j];
          ++sample_count;
        }
      }

      // copy control samples to GPU
      copy_data_to_device_float(status, num_samples, 1, host_control_heading, device_control_heading);
      copy_data_to_device_float(status, num_samples, 1, host_control_gas, device_control_gas);
      copy_data_to_device_float(status, num_samples, 1, host_control_break, device_control_break);

      std::cout << "Using " << sample_count << " control samples." << std::endl;
      std::cout << "Using " << num_blocks << " CUDA block(s)." << std::endl;

    }

    bool mpmi_cb(race_car_shared_control::Mpmi::Request &req, race_car_shared_control::Mpmi::Response &res) {

      // get state
      host_cur_state[0] = req.x;
      host_cur_state[1] = req.y;
      host_cur_state[2] = req.angle;
      host_cur_state[3] = req.x_dot;
      host_cur_state[4] = req.y_dot;
      host_cur_state[5] = req.angle_dot;
      host_cur_state[6] = req.u_1;
      host_cur_state[7] = req.u_2;
      host_cur_state[8] = req.u_3;

      // update current road poly idx, -1 if not in safe region
      road_poly_idx = -1;
      compute_current_road_poly(block_size, num_blocks, host_cur_state[0], host_cur_state[1], road_poly.size(), point_size, num_points_in_poly, poly_size, device_road_poly, device_current_poly);
      copy_data_to_host_int(status, road_poly.size(), 1, host_current_poly, device_current_poly);
      for (int i = 0; i < road_poly.size(); i++) {
        if (host_current_poly[i] == 1) {
          road_poly_idx = i;
        }
      }

      // set up projected states array (in Koopman space)
      int tmp_counter = 0;
      for (int i = 0; i < num_samples*dim_koopman; i+=dim_koopman) {
        // get sampled control and add to state
        host_cur_state[6] = host_control_heading[tmp_counter];
        host_cur_state[7] = host_control_gas[tmp_counter];
        host_cur_state[8] = host_control_break[tmp_counter];
        // project using basis
        host_cur_state_projected = koopman_operator->basis->project(host_cur_state);
        // add to states array
        std::copy(host_cur_state_projected, host_cur_state_projected + dim_koopman, host_states + i);
        tmp_counter += 1;
      }

      // copy states to gpu
      copy_data_to_device_float(status, dim_koopman, num_samples, host_states, device_states);

      // re-initialize safety array and point in poly array
      std::fill(host_safe, host_safe+num_samples, 1);
      copy_data_to_device_int(status, num_samples, 1, host_safe, device_safe);

      // perform forward prediction and safety check on GPU
      for (int i = 0; i < time_horizon; i++) {

        // forward predict
        forward_predict(status, handle, dim_koopman, num_samples, dim_koopman, alpha, beta, device_koopman, device_states);

        // update koopman space state representation
        generate_samples_gpu(block_size, num_blocks, num_samples, dim_koopman, device_control_heading, device_control_gas, device_control_break, device_states);

        // perform safety check
        copy_data_to_device_int(status, num_samples, 1, host_point_in_poly, device_point_in_poly); // reset points in poly to false (0)
        int tmp_poly_idx = road_poly_idx;
        for (int t = 0; t < num_road_poly_check; t++) { // for all polygons to check
          in_poly_gpu(block_size, num_blocks, tmp_poly_idx, point_size, poly_size, num_points_in_poly, device_road_poly, device_states, dim_koopman, num_samples, device_point_in_poly);
          tmp_poly_idx = ((tmp_poly_idx + 1 >= road_poly.size()) ? 0 : tmp_poly_idx + 1);
        }
        update_safe_metric(block_size, num_blocks, num_samples, device_point_in_poly, device_safe);

      }

      // copy safety array back
      copy_data_to_host_int(status, num_samples, 1, host_safe, device_safe);

      // compute optimality of control inputs
      compute_optimality_metric(block_size, num_blocks, req.u_1, req.u_2, req.u_3, device_control_heading, device_control_gas, device_control_break, num_samples, device_safe, device_optimality_metric);
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

      // set return value
      int pick_opt_solution = 0;
      if (num_samples < pick_opt_solution) {
        pick_opt_solution = num_samples;
      }
      if (percent_safe < 0.0000001) { // if we can't find any safe options, just let the user do what they want
        res.u_1 = req.u_1;
        res.u_2 = req.u_2;
        res.u_3 = req.u_3;
        res.percent_safe = 0.0;
        res.cost = 1000000.0;
      } else {
        res.u_1 = host_control_heading[sorted_optimality_metric_indices[optimal_index]];
        res.u_2 = host_control_gas[sorted_optimality_metric_indices[optimal_index]];
        res.u_3 = host_control_break[sorted_optimality_metric_indices[optimal_index]];
        res.percent_safe = percent_safe;
        res.cost = best_optimality_measure;
      }

      return true;
    }

    float* linspace(float a, float b, int n) {
        float* array = new float[n]();
        float step = (b-a) / (n-1);
        for (int i = 0; i < n; i++) {
          array[i] = a;
          a += step;
        }
        return array;
    }

    void road_poly_cb(const race_car_shared_control::RoadPoly::ConstPtr &msg){

      road_poly.clear();
      std::vector<Point> road_seg_poly;
      for (int i = 0; i < msg->polygons.size(); i++) {
        for (int j = 0; j < msg->polygons[i].points.size(); j++) {
          road_seg_poly.push_back(Point(msg->polygons[i].points[j].x, msg->polygons[i].points[j].y));
        }
        road_poly.push_back(road_seg_poly);
        road_seg_poly.clear();
      }

      host_road_poly = new float[point_size * num_points_in_poly * road_poly.size()]();
      cudaMalloc((void**)&device_road_poly, point_size * num_points_in_poly * road_poly.size() * sizeof(float));

      int counter = 0;
      for (int i = 0; i < msg->polygons.size(); i++) {
        for (int j = 0; j < msg->polygons[i].points.size(); j++) {
          host_road_poly[counter] = msg->polygons[i].points[j].x;
          host_road_poly[counter+1] = msg->polygons[i].points[j].y;
          counter += point_size;
        }
      }

      copy_data_to_device_float(status, point_size * num_points_in_poly * road_poly.size(), 1, host_road_poly, device_road_poly);

      host_current_poly = new int[road_poly.size()]();
      cudaMalloc((void**)&device_current_poly, road_poly.size() * sizeof(int));
      copy_data_to_device_int(status, road_poly.size(), 1, host_current_poly, device_current_poly);

    }

    void reset_cb(const std_msgs::String::ConstPtr& msg) {

      delete [] host_road_poly;
      delete [] host_current_poly;
      cudaFree(device_road_poly);
      cudaFree(device_current_poly);
      road_poly_idx = init_road_poly_idx;

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

  ros::init(argc, argv,"model_predictive_control");
  ros::NodeHandle nh;

  ros::Rate loop_rate(20);
  ModelPredictiveControl sys(&loop_rate);

  while (ros::ok()) {
    loop_rate.sleep();
    ros::spinOnce();
  }

  return 0;
}

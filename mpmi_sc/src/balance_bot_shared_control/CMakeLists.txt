cmake_minimum_required(VERSION 2.8.12)
project(balance_bot_shared_control)

add_definitions(-std=c++11)

FIND_PACKAGE(CUDA REQUIRED)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  message_generation
  geometry_msgs
  visualization_msgs
  tf
  cmake_modules
)

add_service_files(
  FILES
  Mpmi.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS
  roscpp
  rospy
  std_msgs
  message_runtime
  geometry_msgs
  visualization_msgs
  cmake_modules
  tf
)

SET(CUDA_NVCC_FLAGS "-arch=compute_50" CACHE STRING "nvcc flags" FORCE)
SET (CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE)
SET(LIB_TYPE STATIC)
CUDA_ADD_LIBRARY(koopman_balance_bot_gpu ${LIB_TYPE} src/koopman_balance_bot_gpu.cu)
add_library(cuda_bb INTERFACE)

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)
include_directories(
  ${catkin_INCLUDE_DIRS}
)

# Model Predictive Controller Node
add_executable(shared_control_balance_bot src/shared_control_balance_bot.cpp)
add_dependencies(shared_control_balance_bot ${catkin_EXPORTED_TARGETS} shared_control_balance_bot_generate_messages_cpp)
cuda_add_cublas_to_target(shared_control_balance_bot)
target_link_libraries(shared_control_balance_bot koopman_balance_bot_gpu ${catkin_LIBRARIES})

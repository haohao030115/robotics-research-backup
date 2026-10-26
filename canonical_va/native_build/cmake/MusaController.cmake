# Alternate build of the original controller; no TensorRT/CUDA dependencies.
find_package(Python3 3.10 REQUIRED COMPONENTS Development.Embed)
find_package(Threads REQUIRED)
find_package(ZLIB REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_library(ZMQ_LIBRARY zmq REQUIRED)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(thirdparty/unitree_sdk2)
add_subdirectory(src/audio_thread)
set(SONIC_CONTROLLER_DIR ${PROJECT_SOURCE_DIR}/src/g1/g1_deploy_onnx_ref)
file(GLOB SONIC_CONTROLLER_SRCS ${SONIC_CONTROLLER_DIR}/src/*.cpp)
add_executable(g1_deploy_onnx_ref ${SONIC_CONTROLLER_SRCS})
target_compile_definitions(g1_deploy_onnx_ref PRIVATE SONIC_MUSA=1 HAS_ROS2=0
  SONIC_MUSA_PYTHON_DIR="${PROJECT_SOURCE_DIR}/scripts")
target_include_directories(g1_deploy_onnx_ref PRIVATE
  ${SONIC_CONTROLLER_DIR}/include ${PROJECT_SOURCE_DIR}/src ${EIGEN3_INCLUDE_DIR})
target_link_libraries(g1_deploy_onnx_ref PRIVATE unitree_sdk2 audio_thread
  ZLIB::ZLIB Python3::Python yaml-cpp ${ZMQ_LIBRARY} Threads::Threads)
set_target_properties(g1_deploy_onnx_ref PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/target/release"
  CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)

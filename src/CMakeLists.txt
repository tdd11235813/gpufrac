cmake_minimum_required(VERSION 2.8)
project(gpufrac)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  if(CUDA_VERSION_STRING VERSION_EQUAL 8.0 OR CUDA_VERSION_STRING VERSION_GREATER 8.0)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 6.0
        OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 6.0)
      message(FATAL_ERROR "CUDA 8.0 does not support gcc6+. Please choose an older gcc version like:\n
cmake -DCMAKE_C_COMPILER=`which gcc-5` -DCMAKE_CXX_COMPILER=`which g++-5` ..")
    endif()
  elseif(CUDA_VERSION_STRING VERSION_EQUAL 7.5 OR CUDA_VERSION_STRING VERSION_GREATER 7.5)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0
        OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 5.0)
      message(FATAL_ERROR "CUDA 7.5 does not support gcc5+. Please choose an older gcc version like:\n
cmake -DCMAKE_C_COMPILER=`which gcc-4` -DCMAKE_CXX_COMPILER=`which g++-4` ..")
    endif()
  endif()
endif()


include(CheckCCompilerFlag)
include(CheckCXXCompilerFlag)

set(CMAKE_CXX_STANDARD 14)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  list(APPEND CMAKE_CXX_FLAGS "-std=gnu++14")
endif ()

find_package(CUDA)
include(FindCUDA)
include_directories(${CUDA_INCLUDE_DIRS})

find_package(OpenGL REQUIRED)
#find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)

include_directories( ${OPENGL_INCLUDE_DIRS} ${GLEW_INCLUDE_DIRS})


set(NANOGUI_BUILD_EXAMPLE OFF CACHE BOOL " " FORCE)
set(NANOGUI_BUILD_SHARED OFF CACHE BOOL " " FORCE)
set(NANOGUI_BUILD_PYTHON OFF CACHE BOOL " " FORCE)
set(NANOGUI_INSTALL OFF CACHE BOOL " " FORCE)
add_definitions(-Wno-ignored-attributes -Wno-deprecated-declarations)
add_subdirectory(
  "${CMAKE_SOURCE_DIR}/ext/nanogui"
  "ext_build/nanogui" #build subdir
)
include_directories(${CMAKE_SOURCE_DIR}/ext/nanogui/include ${NANOGUI_EXTRA_INCS})
link_directories(${PROJECT_BINARY_DIR}/ext_build/nanogui/)

# CUDPP

set(CMAKE_DISABLE_IN_SOURCE_BUILD OFF)
set(CMAKE_DISABLE_SOURCE_CHANGES  OFF)
set(CUDPP_GENCODE_SM20 OFF CACHE BOOL " " FORCE)
set(CUDPP_GENCODE_SM30 ON CACHE BOOL " " FORCE)

list(APPEND CMAKE_MODULE_PATH
  ${CMAKE_SOURCE_DIR}/ext/cudpp/cmake
  )

set(cudpp_INCLUDE_DIRS
  ${CMAKE_SOURCE_DIR}/ext/cudpp/include
  ${CMAKE_SOURCE_DIR}/ext/cudpp/ext/moderngpu/include/
  ${CMAKE_SOURCE_DIR}/ext/cudpp/ext/cub/
  )
include_directories(${cudpp_INCLUDE_DIRS})

# fix cudpp cmake config (run once if files cannot be found)
# execute_process(COMMAND sed -i "s/CMAKE_SOURCE_DIR\\}/CMAKE_CURRENT_SOURCE_DIR\\}\\/\\.\\.\\/\\.\\./g" ${CMAKE_SOURCE_DIR}/ext/cudpp/src/cudpp/CMakeLists.txt)

list(APPEND CUDA_NVCC_FLAGS "-std=c++11 -gencode arch=compute_30,code=compute_30") #--ftz=false") # || --use_fast_math
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

add_subdirectory("${CMAKE_SOURCE_DIR}/ext/cudpp" "ext_build/cudpp")
link_directories(${PROJECT_BINARY_DIR}/ext_build/cudpp/lib)

set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
set(CMAKE_DISABLE_SOURCE_CHANGES  ON)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  list(APPEND CUDA_NVCC_FLAGS "-g -lineinfo")
endif()

cuda_add_executable(${PROJECT_NAME} cuda_frac.cu mccabe.cu main.cpp application.cpp image.cpp shader.cpp)
target_link_libraries(${PROJECT_NAME}
  ${CUDA_LIBRARIES} ${OPENGL_LIBRARIES}
  nanogui ${NANOGUI_EXTRA_LIBS}
  ${GLEW_LIBRARY}
  cudpp
)
  #${GLUT_LIBRARY} ${GLEW_LIBRARY}

add_custom_command(TARGET ${PROJECT_NAME}  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
  ${PROJECT_SOURCE_DIR}/shaders
  $<TARGET_FILE_DIR:${PROJECT_NAME}>/shaders)

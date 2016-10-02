cmake_minimum_required(VERSION 2.8)
project(gpufrac)

include(CheckCCompilerFlag)
include(CheckCXXCompilerFlag)

set(CMAKE_CXX_STANDARD 14)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  list(APPEND CMAKE_CXX_FLAGS "--std=gnu++14")
endif ()

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

find_package(CUDA)
include(FindCUDA)
include_directories(${CUDA_INCLUDE_DIRS})

find_package(OpenGL REQUIRED)
#find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)

include_directories( ${OPENGL_INCLUDE_DIRS} ${GLEW_INCLUDE_DIRS})
# ${GLEW_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS} )

list(APPEND CUDA_NVCC_FLAGS "-std=c++11 -gencode arch=compute_30,code=compute_30") #--ftz=false") # || --use_fast_math

set(CUDA_PROPAGATE_HOST_FLAGS OFF)

cuda_add_executable(${PROJECT_NAME} cuda_frac.cu main.cpp application.cpp image.cpp shader.cpp)
target_link_libraries(${PROJECT_NAME}
  ${CUDA_LIBRARIES} ${OPENGL_LIBRARIES}
  nanogui ${NANOGUI_EXTRA_LIBS}
  ${GLEW_LIBRARY}
)
  #${GLUT_LIBRARY} ${GLEW_LIBRARY}

add_custom_command(TARGET ${PROJECT_NAME}  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
  ${PROJECT_SOURCE_DIR}/shaders
  $<TARGET_FILE_DIR:${PROJECT_NAME}>/shaders)

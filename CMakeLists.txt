cmake_minimum_required(VERSION 2.8)

project(sketch_modeling)


# BOOST
find_package(Boost COMPONENTS system filesystem REQUIRED)

# OpenCV
find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
    message("-- cv @ ${OpenCV_LIBRARIES}")
    include_directories(${OpenCV_INCLUDE_DIRS})
endif(OpenCV_FOUND)

# local dependencies
include_directories(${CMAKE_SOURCE_DIR})
# include_directories(${CMAKE_SOURCE_DIR}/external/)
include_directories(${CMAKE_SOURCE_DIR}/external/eigen3)

add_subdirectory(./monster)

add_executable(rec main.cpp)
target_link_libraries(rec monster "${OpenCV_LIBRARIES}" "${Boost_LIBRARIES}" "${TORCH_LIBRARIES}")
# target_link_libraries(predict "${OpenCV_LIBRARIES}" "${Boost_LIBRARIES}" "${TORCH_LIBRARIES}")



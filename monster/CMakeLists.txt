
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")
list(APPEND CMAKE_CXX_FLAGS "-std=c++11 -fpermissive -fopenmp")

if (POLICY CMP0072)
  set(OpenGL_GL_PREFERENCE LEGACY)
endif()

# OpenCV
find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
endif(OpenCV_FOUND)

# GLM
find_package(GLM REQUIRED)
message(STATUS "GLM included at ${GLM_INCLUDE_DIR}")

# GLFW
find_package(GLFW3 REQUIRED)
message(STATUS "Found GLFW3 in ${GLFW3_INCLUDE_DIR}")

# OpenGL
find_package(OpenGL REQUIRED)
    add_definitions(${OPENGL_DEFINITIONS})
find_package(X11 REQUIRED) # for linux

# OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    message("-- OpenMP ${OpenMP_CXX_LIBRARIES}")
endif()

add_library(GLAD "glad/glad.c")

set(LIBS ${GLFW3_LIBRARY} X11 Xrandr Xinerama Xi Xxf86vm Xcursor GL pthread GLAD)

# set (CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_CXX_LINK_EXECUTABLE} -ldl")

include_directories(./)

file(GLOB monster_src ./*.cpp)

add_library(monster ${monster_src})
target_link_libraries(monster ${LIBS} dl "${OpenCV_LIBRARIES}" "${Boost_LIBRARIES}" "${OpenMP_CXX_LIBRARIES}")
target_include_directories(monster PUBLIC ./)

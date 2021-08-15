cmake_minimum_required(VERSION 3.4)

project(python_dpcpp_module)

set(CMAKE_CXX_COMPILER dpcpp)

find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc highgui)
find_package(PythonInterp 3.6 REQUIRED)
find_package(PythonLibs "${PYTHON_VERSION_STRING}" EXACT REQUIRED)

execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE NUMPY_NOT_FOUND
)
if(NUMPY_NOT_FOUND)
    message(FATAL_ERROR "NumPy headers not found")
endif()

set(target_name python_dpcpp_module)
add_library(${target_name} SHARED dpcpp_module.cpp)

target_include_directories(${target_name} PUBLIC ${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIR})
target_link_libraries(${target_name} ${PYTHON_LIBRARIES} opencv_core opencv_imgproc opencv_highgui)
if(WIN32)
    set_target_properties(${target_name} PROPERTIES SUFFIX ".pyd")
endif()

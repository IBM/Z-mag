cmake_minimum_required(VERSION 3.18)
project(BEM_Assembly LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
if(CMAKE_SYSTEM_PROCESSOR MATCHES "ppc64|powerpc|aix")
    # PowerPC architecture detected
    set(ARCH_FLAGS "-mcpu=native")
else()
    # Default to x86 or others
    set(ARCH_FLAGS "-march=native")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 ${ARCH_FLAGS} -fopenmp")

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Option to build Python bindings
option(BUILD_PYTHON_BINDINGS "Build Python bindings using pybind11" ON)

# Find Python and pybind11 if building bindings
if(BUILD_PYTHON_BINDINGS)
    find_package(Python3 COMPONENTS Interpreter Development QUIET)
    if(Python3_FOUND)
        # Try to find pybind11
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
            OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        
        if(PYBIND11_CMAKE_DIR)
            list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_CMAKE_DIR})
            find_package(pybind11 QUIET)
        endif()
        
        if(NOT pybind11_FOUND)
            message(STATUS "pybind11 not found. Python bindings will not be built.")
            message(STATUS "  Install with: pip install pybind11")
            set(BUILD_PYTHON_BINDINGS OFF)
        endif()
    else()
        message(STATUS "Python3 not found. Python bindings will not be built.")
        set(BUILD_PYTHON_BINDINGS OFF)
    endif()
endif()

# Executable for standalone BEM (CPU-only, no MPI)
add_executable(standalone_BEM
    src/standalone_BEM.cpp
)

# Link math library
target_link_libraries(standalone_BEM m)

# Install target
install(TARGETS standalone_BEM DESTINATION bin)

# Python bindings module (production - direct pointer API)
if(BUILD_PYTHON_BINDINGS AND pybind11_FOUND)
    pybind11_add_module(bem_cpp src/python_bindings.cpp)
    target_link_libraries(bem_cpp PRIVATE m)
    
    # Install to Python site-packages
    install(TARGETS bem_cpp
            LIBRARY DESTINATION ${Python3_SITEARCH})
    
    message(STATUS "Python bindings will be built: bem_cpp (production)")
    
    # Python bindings module (debug - copy-based with validation)
    pybind11_add_module(bem_cpp_debug src/python_bindings_debug.cpp)
    target_link_libraries(bem_cpp_debug PRIVATE m)
    
    # Install to Python site-packages
    install(TARGETS bem_cpp_debug
            LIBRARY DESTINATION ${Python3_SITEARCH})
    
    message(STATUS "Python bindings (debug) will be built: bem_cpp_debug")
endif()

# Print configuration
message(STATUS "")
message(STATUS "========================================")
message(STATUS "  BEM Assembly Configuration")
message(STATUS "========================================")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  CXX flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "========================================")
message(STATUS "")
message(STATUS "Targets:")
message(STATUS "  - standalone_BEM (CPU-only, no MPI)")
if(BUILD_PYTHON_BINDINGS AND pybind11_FOUND)
    message(STATUS "  - bem_cpp (Python module - production, direct pointer API)")
    message(STATUS "  - bem_cpp_debug (Python module - debug, copy-based)")
endif()
message(STATUS "")
message(STATUS "Note: This build uses header-only libraries")
message(STATUS "  - data_loader.hpp (data loading)")
message(STATUS "  - bem_assembler.hpp (BEM computation)")
if(BUILD_PYTHON_BINDINGS AND pybind11_FOUND)
    message(STATUS "  - python_bindings.cpp (Python interface - production)")
    message(STATUS "  - python_bindings_debug.cpp (Python interface - debug)")
endif()
message(STATUS "========================================")
message(STATUS "")

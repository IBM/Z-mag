#!/bin/bash
# Build script for BEM solver

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Change to the script directory
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}BEM Solver Build Script${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Script directory: ${SCRIPT_DIR}"
echo -e "Working directory: $(pwd)"

# Parse command line arguments
BUILD_TYPE="Release"
CLEAN=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --clean)
            CLEAN=1
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --debug      Build in debug mode (default: Release)"
            echo "  --clean      Clean build directory before building"
            echo "  --help       Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Check for required tools
echo -e "\n${YELLOW}Checking prerequisites...${NC}"

if ! command -v cmake &> /dev/null; then
    echo -e "${RED}Error: cmake not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} CMake found: $(cmake --version | head -n1)"

# Verify we're in the right directory
if [ ! -f "CMakeLists.txt" ]; then
    echo -e "${RED}Error: CMakeLists.txt not found in ${SCRIPT_DIR}${NC}"
    echo "This script must be located in the same directory as CMakeLists.txt"
    exit 1
fi

if [ ! -d "src" ]; then
    echo -e "${RED}Error: src/ directory not found${NC}"
    exit 1
fi

# Configure
echo -e "\n${YELLOW}Configuring build...${NC}"
echo "  Source directory: $SCRIPT_DIR"
echo "  Build type: $BUILD_TYPE"

CMAKE_ARGS="-DCMAKE_BUILD_TYPE=$BUILD_TYPE"

# Create build directory
if [ $CLEAN -eq 1 ] && [ -d "build" ]; then
    echo -e "\n${YELLOW}Cleaning build directory...${NC}"
    rm -rf build
fi

mkdir -p build
cd build

# Configure with CMake
cmake .. $CMAKE_ARGS

if [ $? -ne 0 ]; then
    echo -e "${RED}Configuration failed!${NC}"
    exit 1
fi

# Build
echo -e "\n${YELLOW}Building...${NC}"
NPROC=$(nproc 2>/dev/null || echo 4)
make -j$NPROC

if [ $? -ne 0 ]; then
    echo -e "${RED}Build failed!${NC}"
    exit 1
fi

# Success message
BUILD_DIR="${SCRIPT_DIR}/build"
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Build completed successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Build directory: ${BUILD_DIR}"
echo ""
echo "Executables created:"
echo "  • ${BUILD_DIR}/standalone_BEM"
echo ""
# Check if Python modules were built
if [ -f "${BUILD_DIR}/bem_cpp"*.so ]; then
    echo "Python modules created:"
    echo "  • ${BUILD_DIR}/bem_cpp*.so (production - direct pointer API)"
    echo "  • ${BUILD_DIR}/bem_cpp_debug*.so (debug - copy-based)"
    echo ""
    echo "To use Python bindings:"
    echo "  import sys"
    echo "  sys.path.insert(0, '${BUILD_DIR}')"
    echo "  import bem_cpp              # or bem_cpp_debug"
    echo ""
fi
echo "To run (from anywhere):"
echo "  ${BUILD_DIR}/standalone_BEM --rank 0 --size 80"
echo ""
echo "Or navigate to build directory:"
echo "  cd ${BUILD_DIR}"
echo "  ./standalone_BEM --rank 0 --size 80"
echo ""
echo "Example (run all ranks):"
echo "  for i in {0..79}; do"
echo "    ${BUILD_DIR}/standalone_BEM --rank \$i --size 80 &"
echo "  done"
echo "  wait"
echo ""


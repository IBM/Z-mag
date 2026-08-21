#!/bin/bash
#
# Install Python bindings for BEM C++ code
#
# Usage:
#   ./install_python_bindings.sh
#
# This script:
#   1. Checks for pybind11
#   2. Builds the C++ code with Python bindings
#   3. Installs the module or creates a symlink

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}BEM C++ Python Bindings Installer${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check Python
echo -e "${YELLOW}Checking Python...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}ERROR: python3 not found${NC}"
    exit 1
fi
PYTHON_VERSION=$(python3 --version)
echo -e "${GREEN}✓${NC} Found: $PYTHON_VERSION"
echo ""

# Check pybind11
echo -e "${YELLOW}Checking pybind11...${NC}"
if python3 -c "import pybind11" 2>/dev/null; then
    PYBIND11_VERSION=$(python3 -c "import pybind11; print(pybind11.__version__)")
    echo -e "${GREEN}✓${NC} Found: pybind11 version $PYBIND11_VERSION"
else
    echo -e "${RED}✗${NC} pybind11 not found"
    echo ""
    echo "Installing pybind11..."
    pip install pybind11
    if [ $? -ne 0 ]; then
        echo -e "${RED}ERROR: Failed to install pybind11${NC}"
        echo "Try: pip install --user pybind11"
        exit 1
    fi
    echo -e "${GREEN}✓${NC} pybind11 installed"
fi
echo ""

# Build
echo -e "${YELLOW}Building C++ code with Python bindings...${NC}"
./build.sh
if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Build failed${NC}"
    exit 1
fi
echo ""

# Check if modules were built
echo "Checking for built modules..."
FOUND_MODULES=0

# Check for bem_cpp (production - direct API)
if ls build/bem_cpp.*.so build/bem_cpp.*.pyd 2>/dev/null | grep -E "bem_cpp\.(cpython|pyd)" | grep -qv "debug"; then
    MODULE_FILE=$(ls build/bem_cpp.*.so build/bem_cpp.*.pyd 2>/dev/null | grep -v debug | head -1)
    echo -e "${GREEN}✓${NC} bem_cpp (production) built: $(basename $MODULE_FILE)"
    FOUND_MODULES=$((FOUND_MODULES + 1))
else
    echo -e "${YELLOW}⚠${NC} bem_cpp not found"
fi

# Check for bem_cpp_debug
if ls build/bem_cpp_debug*.so build/bem_cpp_debug*.pyd 2>/dev/null | grep -q "bem_cpp_debug"; then
    MODULE_FILE=$(ls build/bem_cpp_debug*.so build/bem_cpp_debug*.pyd 2>/dev/null | head -1)
    echo -e "${GREEN}✓${NC} bem_cpp_debug (debug) built: $(basename $MODULE_FILE)"
    FOUND_MODULES=$((FOUND_MODULES + 1))
else
    echo -e "${YELLOW}⚠${NC} bem_cpp_debug not found"
fi

if [ $FOUND_MODULES -eq 0 ]; then
    echo -e "${RED}ERROR: No Python modules found in build/${NC}"
    echo ""
    echo "This might happen if:"
    echo "  - pybind11 was not found during CMake configuration"
    echo "  - BUILD_PYTHON_BINDINGS was set to OFF"
    exit 1
fi

echo -e "${GREEN}Found $FOUND_MODULES module(s)${NC}"
echo ""

# Ask user about installation
echo -e "${YELLOW}Installation options:${NC}"
echo "  1) Add to PYTHONPATH (recommended for development)"
echo "  2) Install to site-packages (system-wide, requires permissions)"
echo "  3) Skip installation (manual import required)"
echo ""
read -p "Choose option [1-3]: " INSTALL_OPTION

case $INSTALL_OPTION in
    1)
        # Add to PYTHONPATH
        echo ""
        echo -e "${GREEN}To use the modules, add this to your Python script:${NC}"
        echo ""
        echo "    import sys"
        echo "    sys.path.insert(0, '$(pwd)/build')"
        echo "    import bem_cpp              # Production (fast, direct API)"
        echo "    import bem_cpp_debug        # Debug (safe, validated)"
        echo ""
        echo "Or set PYTHONPATH:"
        echo ""
        echo "    export PYTHONPATH=$(pwd)/build:\$PYTHONPATH"
        echo ""
        ;;
    2)
        # Install to site-packages
        echo ""
        echo "Installing to site-packages..."
        cd build
        python3 -m pip install .
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓${NC} Installed successfully"
            echo ""
            echo "You can now use:"
            echo "    import bem_cpp              # Production (fast, direct API)"
            echo "    import bem_cpp_debug        # Debug (safe, validated)"
        else
            echo -e "${RED}ERROR: Installation failed${NC}"
            echo "Try: python3 -m pip install --user ."
            exit 1
        fi
        cd ..
        ;;
    3)
        # Skip
        echo ""
        echo -e "${YELLOW}Skipping installation${NC}"
        echo "Module locations:"
        [ -f "build/bem_cpp"*.so ] || [ -f "build/bem_cpp.cpython"*.so ] && echo "  - bem_cpp (production)"
        [ -f "build/bem_cpp_debug"*.so ] && echo "  - bem_cpp_debug (debug)"
        ;;
    *)
        echo -e "${RED}Invalid option${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Installation Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Verification
echo "Verifying installation..."
VERIFIED=0

if [ "$INSTALL_OPTION" == "2" ]; then
    # Installed globally
    if python3 -c "import bem_cpp; print(f'bem_cpp version: {bem_cpp.__version__}')" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} bem_cpp imports successfully"
        VERIFIED=$((VERIFIED + 1))
    fi
    if python3 -c "import bem_cpp_debug; print(f'bem_cpp_debug version: {bem_cpp_debug.__version__}')" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} bem_cpp_debug imports successfully"
        VERIFIED=$((VERIFIED + 1))
    fi
else
    # Local path
    if python3 -c "import sys; sys.path.insert(0, '$(pwd)/build'); import bem_cpp; print(f'bem_cpp version: {bem_cpp.__version__}')" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} bem_cpp imports successfully"
        VERIFIED=$((VERIFIED + 1))
    fi
    if python3 -c "import sys; sys.path.insert(0, '$(pwd)/build'); import bem_cpp_debug; print(f'bem_cpp_debug version: {bem_cpp_debug.__version__}')" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} bem_cpp_debug imports successfully"
        VERIFIED=$((VERIFIED + 1))
    fi
fi

if [ $VERIFIED -eq 0 ]; then
    echo -e "${RED}✗${NC} No modules could be imported"
    exit 1
fi

echo ""
echo "Next steps:"
echo "  1. Production:            python ../standalone_BEM_cpp.py"
echo "  2. Debug/Development:     python ../standalone_BEM_cpp_debug.py"
echo "  3. Run with MPI:          mpirun -n 80 python ../standalone_BEM_cpp.py"
echo ""
echo "Performance:"
echo "  • bem_cpp (production):   Direct pointer API (fastest, but requires care)"
echo "  • bem_cpp_debug:          Copy-based (slower for large problems, but safe)"
echo ""

